import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch

from ingredient_classifier import (
    DatamoduleParams,
    ExperimentConfig,
    ImgClassifierParams,
    ModelType,
    TrainerConfig,
)


class TestLitImageClassifierModule(unittest.TestCase):
    def setUp(self):
        """Setup test environment with experiment configuration"""
        self.cfg = ExperimentConfig(
            trainer_config=TrainerConfig(max_epochs=1, is_debug=True),
            module_config=ImgClassifierParams(),
            datamodule_config=DatamoduleParams(batch_size=2),
        )
        self.trainer, self.module, self.datamodule = self.cfg.setup_target()
        self.datamodule.setup("fit")

    def test_forward_pass(self):
        """Test forward pass through different model architectures"""
        for model_type in ModelType:
            with self.subTest(model=model_type):
                # Reconfigure with new model type
                self.cfg.module_config.model = model_type
                trainer, module, datamodule = self.cfg.setup_target()
                datamodule.setup("fit")

                # Get a batch and perform forward pass
                batch = next(iter(datamodule.train_dataloader()))
                y_hat = module(batch[0])

                # Verify output dimensions
                self.assertEqual(y_hat.shape[1], self.cfg.module_config.num_classes)
                self.assertEqual(y_hat.shape[0], self.cfg.datamodule_config.batch_size)
                self.assertTrue(torch.is_floating_point(y_hat))

    def test_training_step(self):
        """Test training step with actual data"""
        batch = next(iter(self.datamodule.train_dataloader()))
        loss = self.module.training_step(batch, 0)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, torch.Size([]))
        self.assertTrue(torch.isfinite(loss))

    def test_validation_step(self):
        """Test validation step with actual data"""
        self.datamodule.setup("validate")
        batch = next(iter(self.datamodule.val_dataloader()))
        self.module.validation_step(batch, 0)

        # Verify metrics computation
        self.assertTrue(hasattr(self.module.val_accuracy, "compute"))
        self.assertTrue(hasattr(self.module.confusion_matrix, "compute"))

    def test_metric_computation(self):
        """Test that metrics are properly computed and logged"""
        # Get a batch of data
        batch = next(iter(self.datamodule.train_dataloader()))
        x, y = batch

        # Forward pass and compute metrics
        y_hat = self.module(x)
        train_acc = self.module.train_accuracy(y_hat, y)
        val_acc = self.module.val_accuracy(y_hat, y)

        # Verify metric values
        self.assertGreaterEqual(train_acc, 0.0)
        self.assertLessEqual(train_acc, 1.0)
        self.assertGreaterEqual(val_acc, 0.0)
        self.assertLessEqual(val_acc, 1.0)

    def test_confusion_matrix_computation(self):
        """Test confusion matrix computation and visualization"""
        # Generate predictions
        self.datamodule.setup("validate")
        batch = next(iter(self.datamodule.val_dataloader()))
        x, y = batch
        y_hat = self.module(x)

        # Update confusion matrix
        self.module.confusion_matrix(y_hat, y)
        conf_matrix = self.module.confusion_matrix.compute()

        # Verify confusion matrix dimensions
        self.assertEqual(
            conf_matrix.shape,
            (self.cfg.module_config.num_classes, self.cfg.module_config.num_classes),
        )

        # Test visualization
        plt.figure(figsize=(10, 10))
        sns.heatmap(conf_matrix.cpu().numpy(), annot=True, fmt="d")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")

        # Save and cleanup
        temp_file = Path("temp_confusion_matrix.png")
        plt.savefig(temp_file)
        plt.close()

        self.assertTrue(temp_file.exists())
        temp_file.unlink()


if __name__ == "__main__":
    unittest.main()
