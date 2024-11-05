from enum import Enum, auto
from typing import Dict, Tuple, Type

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch.transforms import ToTensorV2
from pydantic import Field
from torch import Tensor

from litutils import BaseConfig


class TransformType(Enum):
    """
    Enumeration for different types of data transformations.
    """

    TRAIN_FROM_SCRATCH = auto()
    TRAIN_FINE_TUNE = auto()
    VALID = auto()


class TransformsConfig(BaseConfig):
    target: Type["Transformations"] = Field(default_factory=lambda: Transformations)

    img_size: int = 224
    rgb_means: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    """ImageNet mean values for RGB channels."""
    rgb_stds: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    """ImageNet standard deviation values for RGB channels."""

    transform_type: TransformType = TransformType.TRAIN_FROM_SCRATCH


class Transformations:
    def __init__(self, config: TransformsConfig):
        """
        Initializes transformations for different training stages.

        Attributes:
            config (TransformsConfig): Configuration containing image size,
                mean, and standard deviation for normalization.
        """
        self.config = config
        self._pipeline = self.get_transform()

    def __call__(
        self, X: np.ndarray, y: np.ndarray, transform_type: TransformType
    ) -> Tuple[Tensor["3, H, W; float32"], Tensor["1; int64"]]:
        """
        Applies the transformation based on the specified TransformType.

        Args:
            X (np.ndarray["H, W, 3; uint8"]): Input image in HWC format, as a numpy array.
            y (np.ndarray["1; int"]): Label associated with the image.

        Returns:
            Tuple[Tensor["3, H, W; float32"], Tensor["1; int64"]]:
                The transformed image as a tensor in CHW format with float32 dtype,
                and the label as an int64 tensor.
        """
        transformed = self._pipeline(image=X)["image"]
        y = torch.from_numpy(y).to(torch.int64).unsqueeze(0)
        return transformed, y

    def _create_transform(self) -> A.Compose:
        """
        Creates the appropriate transformation pipeline based on the transform_type.

        Returns:
            A.Compose: The composed transformation pipeline.
        """
        match self.config.transform_type:
            case TransformType.TRAIN_FROM_SCRATCH:
                return self._train_from_scratch_transform()
            case TransformType.TRAIN_FINE_TUNE:
                return self._train_fine_tune_transform()
            case TransformType.VALID:
                return self._valid_transform()
            case _:
                raise ValueError(
                    f"Unknown transform type: {self.config.transform_type}"
                )

    def _train_from_scratch_transform(self) -> A.Compose:
        """
        Transformation pipeline for training from scratch.

        Returns:
            A.Compose: The composed transformation pipeline including augmentations
                and normalization suitable for training from scratch.
        """
        return A.Compose(
            [
                A.RandomResizedCrop(
                    height=self.config.img_size,
                    width=self.config.img_size,
                    scale=(0.08, 1.0),
                    ratio=(0.75, 1.3333),
                    p=1.0,
                ),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8
                ),
                A.ToGray(p=0.1),
                A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                A.Normalize(mean=self.config.rgb_means, std=self.config.rgb_stds),
                ToTensorV2(),
            ]
        )

    def _train_fine_tune_transform(self) -> A.Compose:
        """
        Transformation pipeline for fine-tuning pre-trained models.

        Returns:
            A.Compose: The composed transformation pipeline including resizing,
                cropping, and normalization suitable for fine-tuning.
        """
        return A.Compose(
            [
                A.Resize(
                    height=self.config.img_size + 32, width=self.config.img_size + 32
                ),
                A.CenterCrop(height=self.config.img_size, width=self.config.img_size),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=self.config.rgb_means, std=self.config.rgb_stds),
                ToTensorV2(),
            ]
        )

    def _valid_transform(self) -> A.Compose:
        """
        Transformation pipeline for validation.

        Returns:
            A.Compose: The composed transformation pipeline including resizing,
                cropping, and normalization suitable for validation.
        """
        return A.Compose(
            [
                A.Resize(
                    height=self.config.img_size + 32, width=self.config.img_size + 32
                ),
                A.CenterCrop(height=self.config.img_size, width=self.config.img_size),
                A.Normalize(mean=self.config.rgb_means, std=self.config.rgb_stds),
                ToTensorV2(),
            ]
        )
