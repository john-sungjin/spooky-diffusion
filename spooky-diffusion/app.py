import datetime
import os
import time
from enum import Enum
from pathlib import Path

import cv2 as cv
import httpx
import numpy as np


class State(Enum):
    WAITING = 1
    COUNTDOWN = 2
    FLASH = 3
    COOLDOWN = 4


MODAL_URL = (
    "https://village-dev--stable-diffusion-xl-app-john-sungjin-dev.modal.run/generate"
)
PICTURES_DIR = Path("pictures")

def normalize_exposure(image):
    ycrcb = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
    y, cr, cb = cv.split(ycrcb)
    y_eq = cv.equalizeHist(y)
    merged = cv.merge([y_eq, cr, cb])
    return cv.cvtColor(merged, cv.COLOR_YCrCb2BGR)


def main():
    os.environ["DISPLAY"] = ":0"
    face_cascade = cv.CascadeClassifier(
        cv.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    stream = None
    try:
        stream = cv.VideoCapture(0)
        cv.namedWindow("stream", cv.WINDOW_NORMAL)
        cv.setWindowProperty("stream", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)


        if not stream.isOpened():
            print("Cannot open camera")
            exit(0)

        fps = int(stream.get(cv.CAP_PROP_FPS))
        print(f"FPS: {fps}")
        print(f"Frame width: {stream.get(cv.CAP_PROP_FRAME_WIDTH)}")
        print(f"Frame height: {stream.get(cv.CAP_PROP_FRAME_HEIGHT)}")

        num_frames = 0
        detected_faces = []
        state = State.WAITING
        countdown_start = 0
        flash_start = 0
        cooldown_start = 0
        while True:
            ret, frame = stream.read()
            num_frames += 1
            if not ret:
                print("Can't receive frame (stream end?). Exiting...")
                break
            frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
            frame = cv.flip(frame, 1)

            height, width, channels = frame.shape

            if state == State.WAITING:
                if np.average(frame) < 50:
                    frame = cv.add(frame, np.array([50.0]))
                if num_frames % 30 == 0:
                    print("Detecting faces...")
                    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                    detected_faces = face_cascade.detectMultiScale(
                        gray, scaleFactor=1.2, minNeighbors=5, minSize=(50, 50), maxSize=(800,800)
                    )

                if len(detected_faces) > 0:
                    state = State.COUNTDOWN
                    detected_faces = []
                cv.imshow("stream", frame)

            elif state == State.COUNTDOWN:
                if np.average(frame) < 50:
                    frame = cv.add(frame, np.array([50.0]))
                # Countdown is 3 seconds; display countdown on screen
                if countdown_start == 0:
                    countdown_start = time.time()
                elif time.time() - countdown_start > 3:
                    countdown_start = 0
                    state = State.FLASH
                else:
                    height, width, channels = frame.shape

                    # countdown text
                    font = cv.FONT_HERSHEY_SIMPLEX
                    font_scale = 8
                    font_thickness = 10
                    (text_width, text_height), _ = cv.getTextSize(
                        str(3 - int(time.time() - countdown_start)),
                        font,
                        font_scale,
                        font_thickness,
                    )
                    x = (width - text_width) // 2
                    y = (height + text_height) // 2
                    cv.putText(
                        frame,
                        str(3 - int(time.time() - countdown_start)),
                        (x, y),
                        font,
                        font_scale,
                        (255, 255, 255),
                        font_thickness,
                    )

                    # square outline
                    cx, cy = width // 2, height // 2
                    top_left = (cx - height // 2, cy - height // 2)
                    bottom_right = (cx + height // 2, cy + height // 2)
                    cv.rectangle(frame, top_left, bottom_right, (255, 0, 0), 4)

                cv.imshow("stream", frame)

            elif state == State.FLASH:
                # Flash for 1 second, then save image
                if flash_start == 0:
                    flash_start = time.time()

                if time.time() - flash_start > 2:
                    flash_start = 0

                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
                    out_dir = PICTURES_DIR / timestamp
                    out_dir.mkdir(parents=True, exist_ok=True)

                    # save cropped picture
                    height, width, channels = frame.shape
                    cx, cy = width // 2, height // 2
                    cropped_frame = frame[
                        cy - height // 2 : cy + height // 2,
                        cx - height // 2 : cx + height // 2,
                    ]
                    cv.imwrite(str(out_dir / "original.jpg"), cropped_frame)

                    # display cropped picture
                    black_frame = np.zeros_like(frame)
                    start_x = (width - height) // 2
                    black_frame[:, start_x : start_x + height] = cropped_frame

                    font = cv.FONT_HERSHEY_SIMPLEX
                    font_scale = 6
                    font_thickness = 8
                    (text_width, text_height), _ = cv.getTextSize(
                        "Generating...", font, font_scale, font_thickness
                    )
                    x = (width - text_width) // 2
                    y = (height + text_height) // 2
                    cv.putText(
                        black_frame,
                        "Generating...",
                        (x, y),
                        font,
                        font_scale,
                        (255, 255, 255),
                        font_thickness,
                    )

                    print("Showing captured image")
                    cv.imshow("stream", black_frame)
                    cv.waitKey(1)

                    print("Sending to modal")
                    cropped_frame = cv.resize(
                        cropped_frame, (768, 768), interpolation=cv.INTER_AREA
                    )
                    # send to modal
                    _, img_encoded = cv.imencode(".jpg", cropped_frame)
                    img_bytes = img_encoded.tobytes()
                    files = {
                        "image_file": ("image.jpg", img_bytes, "image/jpeg"),
                    }
                    data = {
                        "prompt": "people, halloween costume, night photo, film grain, polaroid",
                        "negative_prompt": "deformed iris, deformed pupils, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, BadDream, UnrealisticDream",
                        "strength": 0.74,
                        "guidance_scale": 10.0,
                        "n_steps": 40,
                    }

                    response = httpx.post(
                        MODAL_URL, files=files, data=data, timeout=None
                    )
                    response.raise_for_status()
                    image_bytes = response.content

                    numpy_array = np.frombuffer(image_bytes, np.uint8)
                    image = cv.imdecode(numpy_array, cv.IMREAD_COLOR)
                    cv.imwrite(str(out_dir / "generated.jpg"), image)

                    # display cropped picture
                    image = cv.resize(
                        image, (height, height), interpolation=cv.INTER_AREA
                    )
                    black_frame = np.zeros_like(frame)
                    start_x = (width - height) // 2
                    black_frame[:, start_x : start_x + height] = image

                    cv.imshow("stream", black_frame)
                    cv.waitKey(1)

                    # pause for 10 seconds
                    time.sleep(10)

                    state = State.COOLDOWN
                else:
                    frame_white = np.ones_like(frame) * np.array(
                        [200, 220, 255], dtype=frame.dtype
                    )
                    cv.imshow("stream", frame_white)
            elif state == State.COOLDOWN:
                if cooldown_start == 0:
                    cooldown_start = time.time()
                elif time.time() - cooldown_start > 10:
                    cooldown_start = 0
                    state = State.WAITING
                cv.imshow("stream", frame)

            if cv.waitKey(1) == ord("q"):
                break
    finally:
        if stream:
            stream.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
