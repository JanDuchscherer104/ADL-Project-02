import unittest
from unittest.mock import MagicMock

from litflow.configs.config import _AbstractExperimentConfig
from pydantic import BaseModel
from pytorch_lightning import LightningDataModule, LightningModule

# Mocking required components


# Mock Module Config
class MyModuleConfig(BaseModel):
    lr: float = 1e-3
    weight_decay: float = 1e-4


# Mock Data Module Config
class MyDataModuleConfig(BaseModel):
    batch_size: int = 32
    num_workers: int = 4


# Mock LightningModule
class MyLightningModule(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, x):
        return x


# Mock LightningDataModule
class MyLightningDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return MagicMock()


# Creating a concrete ExperimentConfig for testing
class MyExperimentConfig(_AbstractExperimentConfig):
    module_config = MyModuleConfig()
    module_type = MyLightningModule
    datamodule_config = MyDataModuleConfig()
    datamodule_type = MyLightningDataModule

    verbose = True
    is_optuna = False
    is_mlflow = True
    max_epochs = 100
    early_stopping_patience = 5
    log_every_n_steps = 10
    is_gpu = True
    is_fast_dev_run = False

    active_callbacks = {
        "ModelCheckpoint": True,
        "TQDMProgressBar": True,
        "EarlyStopping": True,
        "BatchSizeFinder": False,
        "LearningRateMonitor": True,
        "ModelSummary": True,
    }
    paths = MagicMock()  # Mock PathConfig
    mlflow_config = MagicMock()  # Mock MLflowConfig


# Unit Test for MyExperimentConfig
class TestMyExperimentConfig(unittest.TestCase):
    def setUp(self):
        self.config = MyExperimentConfig()

    def test_module_initialization(self):
        # Test module type instantiation
        module = self.config.module_type(self.config.module_config)
        self.assertIsInstance(module, LightningModule)
        self.assertEqual(module.config.lr, 1e-3)
        self.assertEqual(module.config.weight_decay, 1e-4)

    def test_datamodule_initialization(self):
        # Test datamodule type instantiation
        datamodule = self.config.datamodule_type(self.config.datamodule_config)
        self.assertIsInstance(datamodule, LightningDataModule)
        self.assertEqual(datamodule.config.batch_size, 32)
        self.assertEqual(datamodule.config.num_workers, 4)

    def test_experiment_config(self):
        # Test basic attributes of experiment config
        self.assertEqual(self.config.is_debug, False)
        self.assertEqual(self.config.verbose, True)
        self.assertEqual(self.config.max_epochs, 100)
        self.assertEqual(self.config.is_mlflow, True)

    def test_mlflow_setup(self):
        # Mock the mlflow setup
        self.config.mlflow_config.setup_mlflow = MagicMock()
        self.config.__setup_mlflow__()
        self.config.mlflow_config.setup_mlflow.assert_called_once()

    def test_callbacks(self):
        # Test callback configuration
        self.assertTrue(self.config.active_callbacks["ModelCheckpoint"])
        self.assertTrue(self.config.active_callbacks["TQDMProgressBar"])
        self.assertTrue(self.config.active_callbacks["EarlyStopping"])
        self.assertFalse(self.config.active_callbacks["BatchSizeFinder"])

    def test_dump_yaml(self):
        # Mock the paths and YAML dumping to test without actual file I/O
        self.config.paths.configs = MagicMock()
        self.config.to_yaml = MagicMock()
        self.config.dump_yaml()
        self.config.to_yaml.assert_called_once_with(self.config.paths.configs)


if __name__ == "__main__":
    unittest.main()
