from enum import Enum, auto
from typing import Optional, Type

import pytorch_lightning as pl
import torch
from pydantic import Field
from torch import nn
from torch.utils.data import DataLoader, Dataset

from litutils.litutils.utils import BaseConfig, Stage


class DatamoduleParams(BaseConfig["LitDataModule"]):
    batch_size: int = 32
    num_workers: int = 4
    dataset: Stage = Field(default_factory=lambda: Stage.TRAIN)


class LitDataModule(pl.LightningDataModule):
    params: DatamoduleParams

    def __init__(self, params: DatamoduleParams):
        super().__init__()
        self.params = params

    def setup(self, stage: Optional[str] = None):
        self.dataset = torch.utils.data.TensorDataset(
            torch.randn(100, 3, 224, 224), torch.randint(0, 10, (100,))
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.params.batch_size,
            num_workers=self.params.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.params.batch_size,
            num_workers=self.params.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.params.batch_size,
            num_workers=self.params.num_workers,
        )
