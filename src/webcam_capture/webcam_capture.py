from datetime import datetime
from pathlib import Path
from typing import Self, Type

import cv2
from pydantic import Field

from litutils.global_configs.paths import PathConfig
from litutils.utils import BaseConfig


class CaptureConfig(BaseConfig):
    output_dir: Path
    camera_index: int = 0
    target: Type["Webcam"] = Field(
        default_factory=lambda: Webcam,
        description="The target callable class for capturing images",
    )


class Webcam:
    def __init__(self, config: CaptureConfig) -> None:
        self.config = config
        self.output_dir = config.output_dir
        self.cap = cv2.VideoCapture(self.config.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Error: Could not open webcam with index {self.config.camera_index}"
            )

    def capture_image(self) -> None:
        print("Press 'c' to capture an image or 'q' to quit.")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Could not read frame from webcam.")
                break

            cv2.imshow("Webcam", frame)
            key = cv2.waitKey(1) & 0xFF  # 0xFF masks the key to 8 bits

            if key == ord("c"):
                self._save_image(frame)
            elif key == ord("q"):
                print("Quitting...")
                break

        cv2.destroyAllWindows()

    def _save_image(self, frame: cv2.VideoCapture) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = (
            (self.output_dir / f"image_{timestamp}")
            .absolute()
            .with_suffix(".jpg")
            .as_posix()
        )
        cv2.imwrite(filename, frame)
        print(f"Image saved: {filename}")

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_):
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    path_config = PathConfig()
    config = CaptureConfig(output_dir=path_config.webcam_captue)
    with config.setup_target() as webcam:
        webcam.capture_image()
