from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import Dataset
from torchvision.utils import make_grid


class DatasetAnalyzer:
    """Analysis tools for classification datasets with visualization focus on notebooks"""

    def __init__(self, dataset: Dataset):
        """Initialize analyzer with dataset"""
        self.dataset = dataset
        self.df = self._build_analysis_df()

        # Set style for all plots
        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = [10, 6]

    @staticmethod
    def calculate_normalization_stats(
        dataset: Dataset, num_samples: int = 1000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate dataset mean and std for normalization

        Args:
            dataset: Dataset to analyze
            num_samples: Number of random samples to use

        Returns:
            Tuple of (mean, std) arrays with shape (num_channels,)
        """
        indices = np.random.choice(len(dataset), min(num_samples, len(dataset)))
        images = []

        for idx in indices:
            img, _ = dataset[idx]
            if isinstance(img, torch.Tensor):
                images.append(img.numpy())
            else:
                images.append(img)

        images = np.stack(images)

        # Calculate per-channel stats
        mean = images.mean(axis=(0, 2, 3))
        std = images.std(axis=(0, 2, 3))

        return mean, std

    def _build_analysis_df(self) -> pd.DataFrame:
        """Convert dataset samples to pandas DataFrame"""
        data = []
        for idx in range(len(self.dataset)):
            img, label = self.dataset[idx]

            # Get source dataset info
            source_idx = (
                self.dataset.valid_samples[idx][0]
                if hasattr(self.dataset, "valid_samples")
                else None
            )
            source = (
                "Freiburg"
                if source_idx == 0
                else "FruitVeg" if source_idx == 1 else "Unknown"
            )

            # Get class name
            class_name = (
                self.dataset.classes[
                    label.item() if isinstance(label, torch.Tensor) else label
                ]
                if hasattr(self.dataset, "classes")
                else str(label)
            )

            # Image statistics
            if isinstance(img, torch.Tensor):
                channels = [img[i].mean().item() for i in range(img.shape[0])]
            else:
                channels = [img[:, :, i].mean() for i in range(img.shape[-1])]

            data.append(
                {
                    "class": class_name,
                    "source": source,
                    "mean_r": channels[0],
                    "mean_g": channels[1],
                    "mean_b": channels[2],
                }
            )

        return pd.DataFrame(data)

    def show_class_distribution(self):
        """Display class distribution plot"""
        plt.figure()
        g = sns.countplot(
            data=self.df,
            y="class",
            order=self.df["class"].value_counts().index,
            hue="source",
        )
        g.set_title("Sample Distribution Across Classes")
        plt.tight_layout()
        plt.show()

    def show_channel_distributions(self):
        """Display RGB channel distributions"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        channels = ["mean_r", "mean_g", "mean_b"]

        for ax, channel in zip(axes, channels):
            sns.kdeplot(data=self.df, x=channel, hue="class", ax=ax)
            ax.set_title(f'{channel.split("_")[1].upper()} Channel Distribution')

        plt.tight_layout()
        plt.show()

    def show_class_statistics(self):
        """Display per-class channel statistics"""
        stats = (
            self.df.groupby("class")
            .agg(
                {
                    "mean_r": ["mean", "std"],
                    "mean_g": ["mean", "std"],
                    "mean_b": ["mean", "std"],
                }
            )
            .round(3)
        )

        return stats

    def summarize(self):
        """Print comprehensive dataset summary"""
        print(f"Total samples: {len(self.df)}")
        print(f"Number of classes: {self.df['class'].nunique()}")
        print("\nClass distribution:")
        print(self.df["class"].value_counts())
        print("\nSource distribution:")
        print(self.df["source"].value_counts())
        print("\nPer-channel statistics:")
        print(self.show_class_statistics())


# Example notebook usage:
"""
dataset = CompositeGroceryDataset(params)
analyzer = DatasetAnalyzer(dataset)

# Get normalization parameters
mean, std = DatasetAnalyzer.calculate_normalization_stats(dataset)
print(f"Dataset mean: {mean}")
print(f"Dataset std: {std}")

# Show interactive visualizations
analyzer.show_class_distribution()
analyzer.show_channel_distributions()

# Get summary statistics
analyzer.summarize()
"""
