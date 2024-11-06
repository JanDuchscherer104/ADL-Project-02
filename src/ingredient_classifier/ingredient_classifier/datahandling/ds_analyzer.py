from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torchvision.utils import make_grid

from .ds_combined_groceries import CombinedGroceryDataset


class DatasetAnalyzer:
    """Analysis tools for classification datasets with visualization focus"""

    def __init__(self, dataset: CombinedGroceryDataset):
        """Initialize analyzer with dataset"""
        self.dataset = dataset

        # Set style for all plots
        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = [12, 6]

    def get_class_distribution(self) -> np.ndarray:
        """Calculate normalized class distribution vector"""
        # Count samples per class
        label_counts = torch.zeros(self.dataset.num_classes)
        for _, label in self.dataset:
            label_val = label.item() if isinstance(label, torch.Tensor) else label
            label_counts[label_val] += 1

        # Normalize
        return (label_counts / len(self.dataset)).numpy()  # type: ignore

    def plot_class_distribution(self, figsize=(12, 6)):
        """Plot normalized class distribution"""
        dist = self.get_class_distribution()

        plt.figure(figsize=figsize)
        plt.bar(range(len(dist)), dist)
        plt.xticks(range(len(dist)), self.dataset.classes, rotation=45, ha="right")
        plt.title("Normalized Class Distribution")
        plt.ylabel("Fraction of Samples")
        plt.tight_layout()
        plt.show()

    def calculate_channel_stats(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Calculate per-dataset channel mean and std for RGB images

        Computes per-image statistics first, then averages across dataset
        """
        assert self.dataset.params.apply_transforms is False
        dataset = self.dataset

        image_means = []
        image_stds = []

        # Sample indices randomly
        indices = torch.randperm(len(dataset))

        for idx in indices:
            img, _ = dataset[idx]

            # Skip grayscale
            if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[0] == 1):
                continue

            # Handle CHW -> HWC conversion
            if len(img.shape) == 3 and img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0))

            # Convert to float if necessary
            if img.dtype == np.uint8 or img.max() > 1.0:
                img = img.astype(np.float32) / 255.0

            # Compute per-image stats (on HWC format)
            img_mean = np.mean(img, axis=(0, 1))  # mean per channel
            img_std = np.std(img, axis=(0, 1))  # std per channel

            image_means.append(img_mean)
            image_stds.append(img_std)

        mean = np.mean(image_means, axis=0)
        std = np.mean(image_stds, axis=0)

        return {"mean": mean, "std": std}
