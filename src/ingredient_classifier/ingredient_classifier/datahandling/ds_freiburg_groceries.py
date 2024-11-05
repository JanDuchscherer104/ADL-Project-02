from pathlib import Path
from typing import Annotated, Dict, List, Optional, Tuple, Type, Union

import cv2
import torch
from pydantic import Field, ValidationInfo, field_validator
from torch import Tensor
from torch.utils.data import Dataset

from litutils import BaseConfig, PathConfig, Stage

from .transforms import TransformsConfig, TransformsType


class FreiburgGroceriesDatasetParams(BaseConfig["FreiburgGroceriesDataset"]):
    """Parameters for FreiburgGroceriesDataset"""

    paths: PathConfig = Field(default_factory=PathConfig)
    data_dir: Path = Field(
        default=Path(".data/freiburg_groceries"),
        description="Path to dataset directory",
    )
    stage: Stage = Field(
        default=Stage.TRAIN, description="Dataset stage (TRAIN/VAL/TEST)"
    )
    transforms_type: TransformsType = TransformsType.TRAIN_FROM_SCRATCH
    transforms_config: Annotated[TransformsConfig, Field(None)]

    target: Type["FreiburgGroceriesDataset"] = Field(
        default_factory=lambda: FreiburgGroceriesDataset
    )

    @field_validator("transforms_config", mode="before")
    @classmethod
    def __init_transforms_config(cls, _, info: ValidationInfo) -> TransformsConfig:
        tf_type = info.data["transforms_type"]
        match info.data["stage"]:
            case Stage.TRAIN:
                assert tf_type in {
                    TransformsType.TRAIN_FROM_SCRATCH,
                    TransformsType.TRAIN_FINE_TUNE,
                }
                return TransformsConfig(transform_type=tf_type)
            case Stage.VAL | Stage.TEST:
                assert tf_type == TransformsType.VAL
                return TransformsConfig(stage=tf_type)


class FreiburgGroceriesDataset(Dataset):
    """PyTorch Dataset for Freiburg Groceries with unified splits"""

    def __init__(self, params: FreiburgGroceriesDatasetParams):
        """Initialize dataset with specified parameters."""
        self.params = params
        self.transforms = params.transforms_config.setup_target()

        # Map stage to split file name
        split_map = {
            Stage.TRAIN: "train.txt",
            Stage.VAL: "val.txt",
            Stage.TEST: "test.txt",
        }

        # Load split file
        split_file = self.params.data_dir / "splits" / split_map[params.stage]
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")

        # Read and clean image paths
        with split_file.open() as f:
            self.samples = [
                line.strip().split() for line in f.readlines() if line.strip()
            ]

        if not self.samples:
            raise ValueError(f"No samples found in {split_file}")

        # Extract paths and labels
        self.image_paths, labels = zip(*self.samples)
        self.labels: List[int] = list(map(int, labels))  # List[int]

        # Get unique classes from paths
        self.classes: List[str] = sorted(
            list(set(map(lambda path: path.split("/")[0], self.image_paths)))
        )
        self.class_to_idx: Dict[str, int] = dict(
            map((lambda x: (x[1], x[0])), enumerate(self.classes))
        )

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """Get image and label for given index."""
        img_path = self.params.data_dir / "images" / self.image_paths[idx]
        label = self.labels[idx]

        # Load and verify image
        if (image := cv2.imread(img_path.as_posix())) is None:
            raise RuntimeError(f"Failed to load image: {img_path}")

        return self.transforms.apply(X=cv2.cvtColor(image, cv2.COLOR_BGR2RGB), y=label)  # type: ignore

    @property
    def num_classes(self) -> int:
        """Return number of classes in dataset."""
        return len(self.classes)
