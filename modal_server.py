from modal import (
    Image,
    Stub,
    asgi_app,
    gpu,
)


def download_models():
    import transformers
    from huggingface_hub import snapshot_download

    # ignore = ["*.bin", "*.onnx_data", "Counterfeit-*", "*.png"]
    # snapshot_download("gsdf/Counterfeit-V2.5", ignore_patterns=ignore)

    snapshot_download(
        "SG161222/Realistic_Vision_V5.1_noVAE", ignore_patterns=["Realistic_Vision_*"]
    )
    snapshot_download(
        "lllyasviel/control_v11p_sd15_inpaint",
        ignore_patterns=["*fp16.safetensors", "*.bin", "*.png"],
    )
    transformers.utils.move_cache()


image = (
    Image.debian_slim()  # type: ignore
    .apt_install(
        "libglib2.0-0", "libsm6", "libxrender1", "libxext6", "ffmpeg", "libgl1", "wget"
    )
    .pip_install(
        "diffusers~=0.21.0",
        "invisible_watermark~=0.2.0",
        "transformers~=4.34.1",
        "accelerate~=0.24.0",
        "safetensors~=0.4.0",
        "mediapipe",
    )
    .run_function(download_models)
)

stub = Stub("stable-diffusion-xl", image=image)


@stub.function(image=image, concurrency_limit=1, gpu=gpu.A10G(), keep_warm=1)  # type: ignore
@asgi_app()
def app():
    import io

    import diffusers
    import mediapipe as mp
    import numpy as np
    import torch
    from diffusers.utils import load_image
    from fastapi import FastAPI, Form, UploadFile
    from fastapi.responses import Response
    from PIL import Image, ImageEnhance

    web_app = FastAPI()

    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.2
    )

    def generate_face_mask(image: Image.Image, padding_factor=0.10):
        image = np.array(image)
        h, w, c = image.shape
        results = face_detection.process(image)
        mask = np.ones((h, w), dtype=np.uint8) * 255
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                x, y, w, h = (
                    int(bboxC.xmin * iw),
                    int(bboxC.ymin * ih),
                    int(bboxC.width * iw),
                    int(bboxC.height * ih),
                )

                # Calculate center of the bounding box
                center_x = x + w // 2
                center_y = y + h // 2

                # Calculate the radius with padding
                radius = int(max(w, h) // 2 * (1 + padding_factor))

                # Create a meshgrid and calculate distance to the center for each point
                y_indices, x_indices = np.ogrid[: image.shape[0], : image.shape[1]]
                distance_map = (x_indices - center_x) ** 2 + (y_indices - center_y) ** 2

                # Create circular mask where distance to the center is less than radius^2
                mask[distance_map <= radius**2] = 0

        return Image.fromarray(mask)

    # def generate_face_mask(image: Image.Image):
    #     image = np.array(image)
    #     h, w, c = image.shape
    #     results = face_detection.process(image)
    #     mask = np.ones((h, w), dtype=np.uint8) * 255
    #     if results.detections:
    #         for detection in results.detections:
    #             bboxC = detection.location_data.relative_bounding_box
    #             ih, iw, _ = image.shape
    #             x, y, w, h = (
    #                 int(bboxC.xmin * iw),
    #                 int(bboxC.ymin * ih),
    #                 int(bboxC.width * iw),
    #                 int(bboxC.height * ih),
    #             )
    #             mask[y : y + h, x : x + w] = 0
    #     return Image.fromarray(mask)

    def make_inpaint_condition(image: Image.Image, image_mask: Image.Image):
        image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
        image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

        assert (
            image.shape[0:1] == image_mask.shape[0:1]
        ), "image and image_mask must have the same image size"
        image[image_mask > 0.5] = -1.0  # set as masked pixel
        image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        return image

    def adjust_brightness_to_target(pil_image, target_brightness=127):
        # Convert PIL image to NumPy array and to HSV
        image_np = np.asarray(pil_image.convert("HSV"))

        # Extract the brightness (Value) channel and calculate its mean
        mean_brightness = np.mean(image_np[..., 2])

        # Calculate brightness adjustment factor
        factor = target_brightness / mean_brightness

        # Use ImageEnhance to adjust brightness
        enhancer = ImageEnhance.Brightness(pil_image)
        adjusted_image = enhancer.enhance(factor)

        return adjusted_image

    load_options = dict(
        torch_dtype=torch.float16,
        use_safetensors=True,
        device_map="auto",
    )

    # sd: StableDiffusionImg2ImgPipeline = StableDiffusionImg2ImgPipeline.from_pretrained(  # type: ignore
    #     "gsdf/Counterfeit-V2.5",
    #     **load_options,
    # )

    # sd: StableDiffusionImg2ImgPipeline = StableDiffusionImg2ImgPipeline.from_pretrained(  # type: ignore
    #     "SG161222/Realistic_Vision_V5.1_noVAE",
    #     **load_options,
    # )

    controlnet: diffusers.ControlNetModel = diffusers.ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_inpaint", **load_options
    )
    sd: (
        diffusers.StableDiffusionControlNetInpaintPipeline
    ) = diffusers.StableDiffusionControlNetInpaintPipeline.from_pretrained(  # type: ignore
        "SG161222/Realistic_Vision_V5.1_noVAE",
        controlnet=controlnet,
        safety_checker=None,
        **load_options,
    )

    @web_app.post("/generate")
    async def main(
        image_file: UploadFile = Form(...),
        prompt: str = Form(...),
        negative_prompt: str = Form(...),
        strength: float = Form(...),
        guidance_scale: float = Form(...),
        n_steps: int = Form(...),
    ):
        image_data = await image_file.read()
        init_image = Image.open(io.BytesIO(image_data))
        init_image = load_image(init_image)
        init_image = adjust_brightness_to_target(init_image, 100)

        mask_image = generate_face_mask(init_image)
        control_image = make_inpaint_condition(init_image, mask_image)

        image = sd(
            prompt=prompt,
            image=init_image,
            mask_image=mask_image,
            control_image=control_image,
            negative_prompt=negative_prompt,
            strength=strength,  # strength of noise added
            guidance_scale=guidance_scale,  # how closely to follow the prompt
            num_inference_steps=n_steps,
        ).images[0]  # type: ignore

        byte_stream = io.BytesIO()
        image.save(byte_stream, format="PNG")
        image_bytes = byte_stream.getvalue()

        # mask_image.save(byte_stream, format="PNG")
        # image_bytes = byte_stream.getvalue()

        # init_image.save(byte_stream, format="PNG")
        # image_bytes = byte_stream.getvalue()

        return Response(content=image_bytes, media_type="image/png")

    return web_app
