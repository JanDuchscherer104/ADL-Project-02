from enum import Enum, auto
from typing import Optional, Type

import pytorch_lightning as pl
import torch
from pydantic import Field
from torch import nn
from torch.utils.data import DataLoader

from litutils import BaseConfig, Stage

from .datahandling.ds_combined_groceries import GroceryDatasetConfig


class DatamoduleParams(BaseConfig["LitDataModule"]):
    batch_size: int = 32
    num_workers: int = torch.get_num_threads()
    train_ds: GroceryDatasetConfig = GroceryDatasetConfig(stage=Stage.TRAIN)
    val_ds: GroceryDatasetConfig = GroceryDatasetConfig(stage=Stage.VAL)
    test_ds: GroceryDatasetConfig = GroceryDatasetConfig(stage=Stage.TEST)

    target: Type["LitDataModule"] = Field(default_factory=lambda: LitDataModule)


class LitDataModule(pl.LightningDataModule):
    config: DatamoduleParams
    train_ds: GroceryDatasetConfig
    val_ds: GroceryDatasetConfig
    test_ds: GroceryDatasetConfig

    def __init__(self, config: DatamoduleParams):
        super().__init__()
        self.config = config

    def setup(self, stage: Optional[str] = None):
        split = Stage.from_str(stage)
        match split:
            case Stage.TRAIN:
                self.train_ds = self.config.train_ds.setup_target()
            case Stage.VAL:
                self.val_ds = self.config.val_ds.setup_target()
            case Stage.TEST:
                self.test_ds = self.config.test_ds.setup_target()
            case _:
                raise ValueError(f"Invalid stage: {stage}")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
        )
