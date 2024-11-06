import unittest

import torch

from ingredient_classifier import *


class TestExperiment(unittest.TestCase):

    def setUp(self):
        # Base config
        self.cfg = ExperimentConfig(
            trainer_config=TrainerConfig(max_epochs=1, is_debug=True),
            module_config=ImgClassifierParams(num_classes=10, batch_size=2),
            datamodule_config=DatamoduleParams(batch_size=2),
        )

    def test_forward_pass(self):
        """Test forward pass through the model"""
        for model_type in ModelType:
            with self.subTest(model=model_type):
                self.cfg.module_config.model = model_type
                trainer, module, datamodule = self.cfg.setup_target()

                # Setup data and do forward pass
                datamodule.setup("fit")
                batch = next(iter(datamodule.train_dataloader()))
                y_hat = module(batch[0])

                # Check output dimensions
                self.assertEqual(y_hat.shape[1], self.cfg.module_config.num_classes)
                self.assertEqual(y_hat.shape[0], self.cfg.module_config.batch_size)

    def test_training_step(self):
        """Test training step"""
        trainer, module, datamodule = self.cfg.setup_target()
        datamodule.setup("fit")
        batch = next(iter(datamodule.train_dataloader()))

        loss = module.training_step(batch, 0)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, torch.Size([]))

    def test_validation_step(self):
        """Test validation step"""
        trainer, module, datamodule = self.cfg.setup_target()
        datamodule.setup("val")
        batch = next(iter(datamodule.val_dataloader()))

        module.validation_step(batch, 0)
        # Validation step returns None but logs metrics

    def test_datamodule_setup(self):
        """Test datamodule setup and dataloaders"""
        trainer, module, datamodule = self.cfg.setup_target()

        datamodule.setup("fit")
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()

        self.assertEqual(train_loader.batch_size, self.cfg.datamodule_config.batch_size)
        self.assertEqual(val_loader.batch_size, self.cfg.datamodule_config.batch_size)

    def test_trainer_callbacks(self):
        """Test trainer callback setup"""
        trainer, _, _ = self.cfg.setup_target()

        callback_types = [type(cb).__name__ for cb in trainer.callbacks]

        # Check essential callbacks exist
        self.assertIn("ModelCheckpoint", callback_types)
        self.assertIn("EarlyStopping", callback_types)
        self.assertIn("LearningRateMonitor", callback_types)

    def test_optimizer_config(self):
        """Test optimizer setup"""
        trainer, module, _ = self.cfg.setup_target()

        optim_config = module.configure_optimizers()
        self.assertIn("optimizer", optim_config)
        self.assertIn("lr_scheduler", optim_config)

        scheduler_config = optim_config["lr_scheduler"]
        self.assertEqual(scheduler_config["interval"], "step")
        self.assertEqual(scheduler_config["monitor"], "val_loss")


if __name__ == "__main__":
    unittest.main()
