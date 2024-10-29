from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable, Type, TypeVar, Union

import cv2
from pydantic import Field

from litutils.global_configs.paths import PathConfig
from litutils.utils import BaseConfig

T = TypeVar("T")


class CaptureConfig(BaseConfig[Callable]):
    output_dir: Path = Field(
        default=None, description="Directory to save captured images"
    )
    target: Callable[["CaptureConfig"], Callable] = Field(
        default_factory=lambda: Webcam,
        description="The target callable class for capturing images",
    )

    def set_paths_from_config(self, path_config: PathConfig):
        self.output_dir = path_config.data / "captured_images"


class Command(Enum):
    CAPTURE = "capture"
    QUIT = "quit"


class WebcamCaptureContext:
    def __init__(self, config: CaptureConfig):
        self.config = config
        self.webcam = self.config.setup_target()(output_dir=self.config.output_dir)

    def execute(self, command: Command):
        match command:
            case Command.CAPTURE:
                self.webcam.capture_image()
            case Command.QUIT:
                self.webcam.release()
            case _:
                print(f"Unknown command: {command}")


class Webcam:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Error: Could not open webcam.")
        self._ensure_output_directory_exists()

    def _ensure_output_directory_exists(self):
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def capture_image(self):
        print("Press 'c' to capture an image or 'q' to quit.")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Could not read frame from webcam.")
                break

            cv2.imshow("Webcam", frame)
            key = cv2.waitKey(1) & 0xFF

            match key:
                case ord("c"):
                    self._save_image(frame)
                case ord("q"):
                    print("Quitting...")
                    break

    def _save_image(self, frame):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"image_{timestamp}.jpg"
        cv2.imwrite(str(filename), frame)
        print(f"Image saved: {filename}")

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    path_config = PathConfig()
    config = CaptureConfig()
    config.set_paths_from_config(path_config)
    context = WebcamCaptureContext(config=config)
    while True:
        user_input = (
            input("Enter command ('capture' to capture image, 'quit' to exit): ")
            .strip()
            .lower()
        )
        try:
            command = Command(user_input)
            context.execute(command)
        except ValueError:
            print(f"Invalid command: {user_input}")
