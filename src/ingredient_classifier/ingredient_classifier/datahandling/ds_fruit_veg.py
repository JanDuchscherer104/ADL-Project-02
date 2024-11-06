from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type

import cv2
from pydantic import Field
from torch import Tensor
from torch.utils.data import Dataset

from litutils import BaseConfig, PathConfig, Stage

from .transforms import Transforms


class FruitVegDatasetParams(BaseConfig["FruitVegDataset"]):
    """Parameters for FruitVegDataset"""

    paths: PathConfig = Field(default_factory=PathConfig)
    stage: Stage = Stage.TRAIN

    target: Type["FruitVegDataset"] = Field(default_factory=lambda: FruitVegDataset)


class FruitVegDataset(Dataset):
    """Dataset for Fruit and Vegetable Image Recognition"""

    def __init__(
        self,
        params: FruitVegDatasetParams,
        transforms: Optional[Transforms] = None,
    ):
        """
        Args:
            params (FruitVegDatasetParams): Dataset parameters
        """
        self.params = params
        self.data_dir = params.paths.data / "fruit-and-vegetable-image-recognition"

        # Set up transforms
        self.transforms = transforms or (lambda *x: x)

        # Get split directory based on stage
        split_mapping = {
            Stage.TRAIN: "train",
            Stage.VAL: "validation",
            Stage.TEST: "test",
        }
        self.split_dir = self.data_dir / split_mapping[params.stage]

        if not self.split_dir.exists():
            raise RuntimeError(f"Split directory not found: {self.split_dir}")

        self.classes: List[str] = sorted(
            [d.name for d in self.split_dir.iterdir() if d.is_dir()]
        )

        self.class_to_idx: Dict[str, int] = {
            cls_name: i for i, cls_name in enumerate(self.classes)
        }
        self.idx_to_class: Dict[int, str] = dict(
            map(reversed, self.class_to_idx.items())  # type: ignore
        )

        # Collect all image paths and their labels
        self.samples: List[Tuple[Path, int]] = []
        for class_dir in self.split_dir.iterdir():
            if class_dir.is_dir():
                class_idx = self.class_to_idx[class_dir.name]
                for img_path in class_dir.glob("*.jp*g"):  # Match both .jpg and .jpeg
                    self.samples.append((img_path, class_idx))
                for img_path in class_dir.glob("*.png"):
                    self.samples.append((img_path, class_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """
        Args:
            idx (int): Index

        Returns:
            tuple: (image, label) where label is a tensor of size (1,)
        """
        img_path, label = self.samples[idx]

        # Load image
        image = cv2.imread(img_path.as_posix())
        if image is None:
            raise RuntimeError(f"Failed to load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return self.transforms(image, label)  # type: ignore
