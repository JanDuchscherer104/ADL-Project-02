import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torchvision
from torch import Tensor

from ingredient_classifier.datahandling.ds_fruit_veg import (
    FruitVegDataset,
    FruitVegDatasetParams,
)
from ingredient_classifier.datahandling.transforms import TransformsType
from litutils import Stage


class TestFruitVegDataset(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.data_dir = Path(".data/fruit-and-vegetable-image-recognition")
        if not self.data_dir.exists():
            raise RuntimeError("Test data not found. Run prepare_datasets.py first!")

    def create_dataset(
        self, stage: Stage, transform_type: TransformsType
    ) -> FruitVegDataset:
        """Helper to create dataset instance"""
        params = FruitVegDatasetParams(
            data_dir=self.data_dir,
            stage=stage,
            transforms_type=transform_type,
        )
        return FruitVegDataset(params)

    def test_dataset_initialization(self):
        """Test basic dataset properties"""
        dataset = self.create_dataset(Stage.TRAIN, TransformsType.TRAIN_FROM_SCRATCH)

        self.assertTrue(len(dataset) > 0)
        self.assertTrue(len(dataset.classes) > 0)
        self.assertEqual(len(dataset.class_to_idx), len(dataset.classes))

    def test_data_loading(self):
        """Test data loading and tensor shapes"""
        dataset = self.create_dataset(Stage.TRAIN, TransformsType.TRAIN_FROM_SCRATCH)

        # Get first sample
        image, label = dataset[0]

        # Check types
        self.assertIsInstance(image, Tensor)
        self.assertIsInstance(label, Tensor)

        # Check shapes
        self.assertEqual(len(image.shape), 3)  # C,H,W
        self.assertEqual(image.shape[0], 3)  # RGB channels
        self.assertEqual(image.shape[1], 224)  # Height
        self.assertEqual(image.shape[2], 224)  # Width
        self.assertEqual(label.shape, torch.Size([1]))

    def test_transforms_visualization(self):
        """Visualize training samples with transforms"""
        dataset = self.create_dataset(Stage.TRAIN, TransformsType.TRAIN_FROM_SCRATCH)

        # Get multiple samples
        n_samples = 16
        images, labels = [], []
        for i in range(n_samples):
            img, lbl = dataset[i]
            images.append(img)
            labels.append(dataset.classes[lbl.item()])

        # Create and save grid
        self._save_image_grid(
            images, "Training Samples with Augmentation", "fruit_veg_train_samples.png"
        )

    def test_transforms_visualization_val(self):
        """Visualize validation samples"""
        dataset = self.create_dataset(Stage.VAL, TransformsType.VAL)

        # Get multiple samples
        n_samples = 16
        images, labels = [], []
        for i in range(n_samples):
            img, lbl = dataset[i]
            images.append(img)
            labels.append(dataset.classes[lbl.item()])

        # Create and save grid
        self._save_image_grid(
            images, "Validation Samples (Center Crop)", "fruit_veg_val_samples.png"
        )

    def _save_image_grid(self, images: list, title: str, filename: str):
        """Helper to create and save image grid"""
        img_grid = torchvision.utils.make_grid(
            images, nrow=4, normalize=True, pad_value=1
        )

        plt.figure(figsize=(15, 15))
        plt.imshow(img_grid.permute(1, 2, 0))
        plt.title(title)
        plt.axis("off")

        save_path = Path("test_output")
        save_path.mkdir(exist_ok=True)
        plt.savefig(save_path / filename)
        plt.close()

    def test_all_stages(self):
        """Test dataset with all stages"""
        stages = [Stage.TRAIN, Stage.VAL, Stage.TEST]

        for stage in stages:
            transform_type = (
                TransformsType.TRAIN_FROM_SCRATCH
                if stage == Stage.TRAIN
                else TransformsType.VAL
            )

            dataset = self.create_dataset(stage, transform_type)

            # Basic checks
            self.assertTrue(len(dataset) > 0)

            # Load sample and check normalization
            image, _ = dataset[0]
            self.assertTrue(torch.all(image >= -5) and torch.all(image <= 5))

    def test_class_consistency(self):
        """Test class mapping consistency across stages"""
        datasets = {
            stage: self.create_dataset(
                stage,
                (
                    TransformsType.TRAIN_FROM_SCRATCH
                    if stage == Stage.TRAIN
                    else TransformsType.VAL
                ),
            )
            for stage in [Stage.TRAIN, Stage.VAL, Stage.TEST]
        }

        # Check class lists are identical
        base_classes = set(datasets[Stage.TRAIN].classes)
        for stage in [Stage.VAL, Stage.TEST]:
            self.assertEqual(base_classes, set(datasets[stage].classes))


if __name__ == "__main__":
    unittest.main()
