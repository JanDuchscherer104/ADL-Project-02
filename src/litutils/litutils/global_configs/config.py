import re
from datetime import datetime
from pathlib import Path
from typing import Annotated, Dict, Literal, Optional, Type

import mlflow
import psutil
from pydantic import Field, ValidationInfo, field_validator, model_validator
from pytorch_lightning import LightningDataModule, LightningModule
from typing_extensions import Self

from ..lightning.lit_datamodule import DatamoduleParams, LitNosmodDatamodule
from ..lightning.lit_module import HParams, LitNOSModModule
from ..utils import CONSOLE, BaseConfig
from .mlflow import MLflowConfig
from .paths import PathConfig


class _GlobalExperimentConfig(BaseConfig):
    """
    TODO: add MachineConfig
    TODO: add TrainerConfig
    """

    module_config: BaseConfig  # To be defined in child classes
    module_type: Type[LightningModule] = Field(
        ...
    )  # Expect a class derived from LightningModule
    datamodule_config: BaseConfig  # To be defined in child classes
    datamodule_type: Type[LightningDataModule] = Field(
        ...
    )  # Expect a class derived from LightningDataModule
    experiments: Dict[Literal["classifier"],]

    is_debug: bool = False
    verbose: bool = True
    from_ckpt: Optional[str] = None
    is_optuna: bool = False
    is_mlflow: bool = False
    max_epochs: int = 50
    early_stopping_patience: int = 2
    log_every_n_steps: int = 8
    is_gpu: bool = True
    matmul_precision: Literal["medium", "high"] = "medium"
    is_fast_dev_run: bool = False
    active_callbacks: Dict[
        Literal[
            "ModelCheckpoint",
            "TQDMProgressBar",
            "EarlyStopping",
            "BatchSizeFinder",
            "LearningRateMonitor",
            "ModelSummary",
        ],
        bool,
    ] = {
        "ModelCheckpoint": True,
        "TQDMProgressBar": True,
        "EarlyStopping": True,
        "BatchSizeFinder": False,
        "LearningRateMonitor": False,
        "ModelSummary": True,
    }
    paths: PathConfig = Field(default_factory=PathConfig)
    mlflow_config: MLflowConfig = Field(default_factory=MLflowConfig)

    def dump_yaml(self) -> None:
        self.to_yaml(self.paths.configs)

    @model_validator(mode="after")
    def __setup_mlflow(self) -> Self:
        if self.is_mlflow:
            self.mlflow_config.setup_mlflow(mlflow_uri=self.paths.mlflow_uri)

        return self

    def dump(self):
        i = 1
        config_file = self.paths.configs / f"{self.mlflow_config.run_name}.yaml"
        while config_file.exists():
            config_file = self.paths.configs / f"{self.mlflow_config.run_name}-{i}.yaml"
            i += 1
        self.to_yaml(config_file)

    @classmethod
    def read(cls, file_name: str, root: Optional[Path] = None) -> Self:
        root = root or Path(__file__).parents[3].resolve()
        config_file = (root / ".configs" / file_name).with_suffix(".yaml")
        assert config_file.exists(), f"{config_file} does not exist"
        return cls.from_yaml(config_file)  # type: ignore
