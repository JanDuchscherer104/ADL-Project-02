import unittest
import warnings
from pathlib import Path

import matplotlib as mpl
import torch
import torchvision.utils as vutils
from matplotlib import pyplot as plt
from torch import Tensor
from torch.utils.data import DataLoader

from ingredient_classifier.datahandling.ds_combined_groceries import (
    CombinedGroceryDataset,
    GroceryDatasetConfig,
)
from ingredient_classifier.datahandling.ds_freiburg_groceries import (
    FreiburgGroceriesDataset,
)
from ingredient_classifier.datahandling.ds_fruit_veg import FruitVegDataset
from ingredient_classifier.datahandling.transforms import TransformsType
from litutils import PathConfig, Stage


class TestCompositeGroceryDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures once for all tests"""
        # Suppress warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", message=".*iCCP.*")

        cls.paths = PathConfig()
        if not (cls.paths.data / "freiburg_groceries").exists():
            raise RuntimeError("Freiburg dataset not found!")
        if not (cls.paths.data / "fruit-and-vegetable-image-recognition").exists():
            raise RuntimeError("Fruit/Veg dataset not found!")

        # Cache datasets for different stages
        cls.datasets = {}
        for stage in [Stage.TRAIN, Stage.VAL, Stage.TEST]:
            transform_type = (
                TransformsType.TRAIN_FROM_SCRATCH
                if stage == Stage.TRAIN
                else TransformsType.VAL
            )
            cls.datasets[stage] = cls._create_dataset(cls.paths, stage, transform_type)

    @staticmethod
    def _create_dataset(
        paths: PathConfig, stage: Stage, transform_type: TransformsType
    ) -> CombinedGroceryDataset:
        """Helper to create dataset instance"""
        params = GroceryDatasetConfig(
            paths=paths,
            stage=stage,
            transforms_type=transform_type,
        )
        return params.setup_target()

    def test_dataset_initialization(self):
        """Test basic dataset properties"""
        dataset = self.datasets[Stage.TRAIN]
        self.assertTrue(len(dataset) > 0)
        self.assertTrue(len(dataset.classes) > 0)
        self.assertEqual(len(dataset.class_to_idx), len(dataset.classes))
        self.assertIn("corn", dataset.classes)
        self.assertNotIn("sweetcorn", dataset.classes)
        self.assertNotIn("candy", dataset.classes)

    def test_data_loading(self):
        """Test data loading and tensor shapes"""
        dataset = self.datasets[Stage.TRAIN]
        image, label = dataset[0]

        self.assertIsInstance(image, Tensor)
        self.assertIsInstance(label, Tensor)
        self.assertEqual(len(image.shape), 3)
        self.assertEqual(image.shape[0], 3)
        self.assertEqual(image.shape[1], 224)
        self.assertEqual(image.shape[2], 224)

    def _visualize_samples(
        self,
        dataset: CombinedGroceryDataset,
        stage: str,
        transform_type: TransformsType,
    ):
        """Helper to create and save visualization grid"""
        plt.rcParams["font.size"] = 14

        loader = DataLoader(
            dataset,
            batch_size=8,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )

        for grid_idx in range(3):
            try:
                images, labels = next(iter(loader))
            except StopIteration:
                break

            # Fix: Get correct sample indices for the batch
            batch_indices = torch.randperm(len(dataset))[: len(labels)]

            # Get source and class info using actual indices
            label_texts = [
                f"{'Freiburg' if dataset.samples[batch_indices[i]][0] == 0 else 'Fruit/Veg'}: {dataset.classes[lbl.item()]}"
                for i, lbl in enumerate(labels)
            ]

            img_grid = vutils.make_grid(images, nrow=4, normalize=True, pad_value=1)
            plt.figure(figsize=(12, 12))
            plt.imshow(img_grid.permute(1, 2, 0))
            plt.title(f"{stage} Samples ({transform_type}) - Grid {grid_idx+1}", pad=20)

            for idx, label in enumerate(label_texts):
                row, col = idx // 4, idx % 4
                plt.text(
                    img_grid.shape[2] * (col * 0.25 + 0.125),
                    img_grid.shape[1] * (row * 0.25 + 0.20),
                    label,
                    color="red",
                    ha="center",
                    va="center",
                    fontsize=16,
                    bbox=dict(facecolor="white", alpha=0.8),
                )

            plt.axis("off")
            save_path = Path("test_output")
            save_path.mkdir(exist_ok=True)
            plt.savefig(
                save_path / f"composite_{stage.lower()}_samples_{grid_idx}.png",
                bbox_inches="tight",
                dpi=150,
            )
            plt.close()

    def test_source_dataset_mapping(self):
        """Test that samples are correctly mapped to their source datasets"""
        dataset = self.datasets[Stage.TRAIN]

        assert isinstance(dataset, CombinedGroceryDataset)

        # Check distribution of sources
        freiburg_count = sum(1 for x in dataset.samples if x[0] == 0)
        fruitveg_count = sum(1 for x in dataset.samples if x[0] == 1)

        self.assertTrue(freiburg_count > 0, "No samples from Freiburg dataset")
        self.assertTrue(fruitveg_count > 0, "No samples from Fruit/Veg dataset")

        # Test random samples
        for _ in range(10):
            idx = torch.randint(len(dataset), (1,)).item()
            dataset_idx, sample_idx, mapped_label = dataset.samples[idx]

            # Get original class from source dataset
            source_dataset = dataset.datasets[dataset_idx]
            assert isinstance(source_dataset, FreiburgGroceriesDataset) or isinstance(
                source_dataset, FruitVegDataset
            )
            _, orig_label = source_dataset[sample_idx]
            orig_class = (
                source_dataset.classes[
                    (
                        orig_label.item()
                        if isinstance(orig_label, torch.Tensor)
                        else orig_label
                    )
                ]
                .replace(" ", "_")
                .lower()
            )

            mapped_class = dataset.classes[mapped_label]

            # Verify mapping
            self.assertEqual(
                mapped_class,
                dataset.CLASS_MAPPING.get(orig_class, orig_class),
                f"Wrong mapping for {orig_class} -> {mapped_class}",
            )

    def test_transforms_visualization(self):
        """Visualize training samples"""
        self._visualize_samples(
            self.datasets[Stage.TRAIN], "Training", TransformsType.TRAIN_FROM_SCRATCH
        )

    def test_transforms_visualization_val(self):
        """Visualize validation samples"""
        self._visualize_samples(
            self.datasets[Stage.VAL], "Validation", TransformsType.VAL
        )

    def test_class_consistency(self):
        """Test class mapping consistency across stages"""
        base_classes = set(self.datasets[Stage.TRAIN].classes)
        for stage in [Stage.VAL, Stage.TEST]:
            self.assertEqual(base_classes, set(self.datasets[stage].classes))

    def test_class_filtering(self):
        """Test proper class filtering"""
        dataset = self.datasets[Stage.TRAIN]
        for cls in dataset.params.classes_to_drop:
            self.assertNotIn(cls, dataset.classes)


if __name__ == "__main__":
    unittest.main()
