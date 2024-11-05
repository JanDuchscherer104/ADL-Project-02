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
        default=Stage.TRAIN, description="Dataset stage (train/val/test)"
    )
    transforms_type: TransformsType = TransformsType.TRAIN_FROM_SCRATCH
    transforms_config: Annotated[TransformsConfig, Field(None)]

    target: Type["FreiburgGroceriesDataset"] = Field(
        default=lambda: FreiburgGroceriesDataset
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
    """PyTorch Dataset for Freiburg Groceries"""

    def __init__(self, params: FreiburgGroceriesDatasetParams):
        """
        Args:
            params: Dataset parameters
        """
        self.params = params
        self.transforms = params.transforms_config.setup_target()

        # Load split files
        stage = "validation" if params.stage == Stage.VAL else params.stage.value[0]
        split_file = self.params.data_dir / "splits" / f"{stage}.txt"
        with split_file.open("r") as f:
            self.image_paths = list(f.readlines())

        # Get class names from image paths
        self.classes = sorted(
            list(set(path.split("/")[0] for path in self.image_paths))
        )
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        img_path = self.params.data_dir / "images" / self.image_paths[idx]
        label = self.class_to_idx[self.image_paths[idx].split("/")[0]]

        # Load and transform image
        image = cv2.imread(img_path.as_posix())

        return self.transforms(X=image, y=label)  # type: ignore

    @property
    def num_classes(self) -> int:
        return len(self.classes)
