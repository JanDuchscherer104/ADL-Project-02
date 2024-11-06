from enum import Enum, auto
from typing import Tuple, Type

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch.transforms import ToTensorV2
from pydantic import Field
from torch import Tensor

from litutils import BaseConfig


class TransformsType(Enum):
    """
    Enumeration for different types of data transformations.
    """

    TRAIN_FROM_SCRATCH = auto()
    TRAIN_FINE_TUNE = auto()
    VAL = auto()
    IDENT = auto()


class TransformsConfig(BaseConfig["Transforms"]):
    target: Type["Transforms"] = Field(default_factory=lambda: Transforms)

    img_size: int = 224
    rgb_means: Tuple[float, float, float] = (
        0.5888754,
        0.53425103,
        0.41462088,
    )  # (0.485, 0.456, 0.406)
    """ImageNet mean values for RGB channels."""
    rgb_stds: Tuple[float, float, float] = (
        0.23366731,
        0.23470095,
        0.24934198,
    )  # (0.229, 0.224, 0.225)
    """ImageNet standard deviation values for RGB channels."""

    transform_type: TransformsType = TransformsType.TRAIN_FROM_SCRATCH


class Transforms:
    def __init__(self, config: TransformsConfig):
        """
        Initializes transformations for different training stages.

        Attributes:
            config (TransformsConfig): Configuration containing image size,
                mean, and standard deviation for normalization.
        """
        self.config = config
        self._pipeline = self._create_transform()

    def __call__(self, X: np.ndarray, y: np.ndarray) -> Tuple[Tensor, Tensor]:
        """
        Applies the transformation based on the specified TransformType.

        Args:
            X (np.ndarray["H, W, 3; uint8"]): Input image in HWC format, as a numpy array.
            y (np.ndarray["1; int"]): Label associated with the image.

        Returns:
            Tuple[Tensor["3, H, W; float32"], Tensor["0; int64"]]:
                The transformed image as a tensor in CHW format with float32 dtype,
                and the label as an int64 tensor.
        """
        return (
            self._pipeline(image=X)["image"],
            torch.tensor(y, dtype=torch.int64),
        )

    def _create_transform(self) -> A.Compose:
        """
        Creates the appropriate transformation pipeline based on the transform_type.

        Returns:
            A.Compose: The composed transformation pipeline.
        """
        match self.config.transform_type:
            case TransformsType.TRAIN_FROM_SCRATCH:
                return self._train_from_scratch_transform()
            case TransformsType.TRAIN_FINE_TUNE:
                return self._train_fine_tune_transform()
            case TransformsType.VAL:
                return self._val_transform()
            case _:
                raise ValueError(
                    f"Unknown transform type: {self.config.transform_type}"
                )

    def _train_from_scratch_transform(self) -> A.Compose:
        return A.Compose(
            [
                A.SmallestMaxSize(max_size=self.config.img_size),
                A.RandomResizedCrop(
                    size=(self.config.img_size, self.config.img_size),
                    scale=(0.08, 1.0),
                    ratio=(0.75, 1.3333),
                    p=1.0,
                ),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                A.CoarseDropout(
                    num_holes_range=(1, 8),
                    hole_height_range=(4, 16),
                    hole_width_range=(4, 16),
                    p=0.5,
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.5
                ),
                A.Perspective(scale=(0.05, 0.1), p=0.5),
                A.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8
                ),
                A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                A.Normalize(mean=self.config.rgb_means, std=self.config.rgb_stds),
                ToTensorV2(),
            ]
        )

    def _train_fine_tune_transform(self) -> A.Compose:
        """Fine-tuning pipeline with lighter augmentations."""
        return A.Compose(
            [
                A.SmallestMaxSize(max_size=self.config.img_size),
                A.RandomResizedCrop(
                    height=self.config.img_size,
                    width=self.config.img_size,
                    scale=(0.8, 1.0),
                    ratio=(0.9, 1.1),
                    p=1.0,
                ),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.1, contrast_limit=0.1, p=0.3
                ),
                A.HueSaturationValue(
                    hue_shift_limit=5, sat_shift_limit=5, val_shift_limit=5, p=0.3
                ),
                A.Normalize(mean=self.config.rgb_means, std=self.config.rgb_stds),
                ToTensorV2(),
            ]
        )

    def _val_transform(self) -> A.Compose:
        """Validation pipeline with deterministic transforms."""
        return A.Compose(
            [
                A.SmallestMaxSize(max_size=self.config.img_size),
                A.CenterCrop(height=self.config.img_size, width=self.config.img_size),
                A.Normalize(mean=self.config.rgb_means, std=self.config.rgb_stds),
                ToTensorV2(),
            ]
        )
