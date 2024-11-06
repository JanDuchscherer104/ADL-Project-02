from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union
from warnings import warn

import torch
from pydantic import Field, model_validator
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import Logger
from pytorch_lightning.utilities.types import STEP_OUTPUT

from litutils import CONSOLE, BaseConfig
from litutils.global_configs.wandb import WandbConfig


class CallbacksConfig(BaseConfig):
    """Active callbacks configuration"""

    model_checkpoint: bool = True
    progress_bar: bool = True
    early_stopping: bool = True
    batch_size_finder: bool = False
    lr_monitor: bool = True
    model_summary: bool = True
    # optuna_pruning: bool = False  # TODO

    # pruning_monitor: str = "val_loss"
    # pruning_mode: Literal["min", "max"] = "min"

    # @model_validator(mode="after")
    # def __check_compatible_callbacks(self) -> None:
    #     """Check for incompatible callback combinations"""
    #     assert not self.optuna_pruning and not self.early_stopping


class TrainerConfig(BaseConfig["Trainer"]):
    """PyTorch Lightning Trainer configuration"""

    # Training params
    max_epochs: int = 50
    early_stopping_patience: int = 2
    log_every_n_steps: int = 8
    fast_dev_run: bool = False
    gradient_clip_val: Optional[float] = 0.5
    accumulate_grad_batches: int = 1
    precision: Union[str, int] = "32-true"
    val_check_interval: Union[int, float] = 1.0

    # Hardware settings
    accelerator: str = "auto"
    """
    Literal
    """
    devices: Union[int, str] = "auto"
    num_workers: int = Field(default_factory=lambda: torch.get_num_threads())
    matmul_precision: Literal["medium", "high"] = "medium"

    # Callback settings
    callbacks: CallbacksConfig = Field(default_factory=CallbacksConfig)

    # Logging
    wandb: WandbConfig = WandbConfig(
        project="IngredientClassifier",
    )

    # Propagated fields
    is_debug: bool = False
    verbose: bool = True

    def setup_target(self, **kwargs: Any) -> Trainer:
        """Setup trainer using factory pattern"""
        if self.is_debug:
            self._setup_debug_mode()
        return TrainerFactory.create(self, **kwargs)

    def _setup_debug_mode(self) -> None:
        """Configure debug settings"""
        self.fast_dev_run = True
        self.accelerator = "cpu"
        self.devices = 1
        self.num_workers = 0
        self.callbacks.model_checkpoint = False
        torch.autograd.set_detect_anomaly(True)


class TrainerFactory:
    """Factory for creating PyTorch Lightning Trainer instances"""

    def __init__(self, config: TrainerConfig):
        self.config = config

    @classmethod
    def create(cls, config: TrainerConfig, **kwargs: Any) -> Trainer:
        """Create and configure a new Trainer instance"""
        factory = cls(config)

        # Setup hardware precision
        torch.set_float32_matmul_precision(config.matmul_precision)

        # Assemble components
        callbacks = factory._assemble_callbacks()
        loggers = factory._assemble_loggers()

        # Create trainer
        trainer = Trainer(
            max_epochs=config.max_epochs,
            accelerator=config.accelerator,
            devices=config.devices,
            log_every_n_steps=config.log_every_n_steps,
            fast_dev_run=config.fast_dev_run,
            gradient_clip_val=config.gradient_clip_val,
            accumulate_grad_batches=config.accumulate_grad_batches,
            precision=config.precision,
            val_check_interval=config.val_check_interval,
            callbacks=callbacks,
            logger=loggers,
            **kwargs,
        )
        return trainer

    def _assemble_callbacks(self) -> List[Callback]:
        """Create and return active callbacks"""
        callbacks: List[Callback] = []

        if self.config.callbacks.model_checkpoint:
            callbacks.append(
                ModelCheckpoint(
                    monitor="val_loss", mode="min", verbose=self.config.verbose
                )
            )

        if self.config.callbacks.early_stopping:
            callbacks.append(
                EarlyStopping(
                    monitor="val_loss",
                    patience=self.config.early_stopping_patience,
                    mode="min",
                    verbose=self.config.verbose,
                )
            )

        if self.config.callbacks.lr_monitor:
            callbacks.append(
                LearningRateMonitor(logging_interval="step", log_momentum=True)
            )

        if self.config.callbacks.model_summary:
            callbacks.append(ModelSummary(max_depth=4))

        if self.config.callbacks.progress_bar:
            callbacks.append(CustomTQDMProgressBar())

        return callbacks

    def _assemble_loggers(self) -> Union[List[Logger], bool]:
        """Setup logging systems"""
        if self.config.wandb:
            return [self.config.wandb.setup_target()]
        return True  # Default logger


class CustomTQDMProgressBar(TQDMProgressBar):
    """Custom progress bar with loss display"""

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        description = f"train_loss: {trainer.callback_metrics.get('train_loss', 0):.2f}"
        self.train_progress_bar.set_postfix_str(description, refresh=True)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        super().on_validation_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )
        description = f"val_loss: {trainer.callback_metrics.get('val_loss', 0):.2f}"
        self.val_progress_bar.set_postfix_str(description, refresh=True)
