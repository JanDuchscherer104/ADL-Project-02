import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
from torch import nn, tensor
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from litutils import WandbConfig


class MockModel(pl.LightningModule):
    def __init__(self):
        super(MockModel, self).__init__()
        self.layer = nn.Linear(1, 1)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = nn.functional.mse_loss(self(x), y)
        self.log("train/loss", loss, logger=True)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.001)

    def train_dataloader(self):
        dataset = TensorDataset(
            tensor([[1.0], [2.0], [3.0]]), tensor([[1.0], [2.0], [3.0]])
        )
        return DataLoader(dataset, batch_size=1)


class TestWandbConfig(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.wandb_config = WandbConfig(
            project="test_project",
            save_dir=self.temp_dir.name,
            log_model=True,
            target=WandbLogger,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_wandb_logger_initialization(self):
        logger = self.wandb_config.setup_target()
        self.assertIsInstance(logger, WandbLogger)
        self.assertEqual(logger.experiment.project_name(), "test_project")

    def test_wandb_logger_with_trainer(self):
        logger = self.wandb_config.setup_target()
        model = MockModel()
        trainer = pl.Trainer(max_epochs=1, logger=logger, log_every_n_steps=1)
        trainer.fit(model)

        log_dir = Path(self.temp_dir.name) / self.wandb_config.project
        self.assertTrue(log_dir.exists())


if __name__ == "__main__":
    unittest.main()
