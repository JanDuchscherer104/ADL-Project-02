from pathlib import Path
from typing import Any, List, Literal, Optional, Self, Union

import torch
import wandb.util
from optuna import Trial
from optuna_integration import PyTorchLightningPruningCallback
from pydantic import Field, model_validator
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import Logger
from pytorch_lightning.utilities.types import STEP_OUTPUT

from litutils import BaseConfig, PathConfig
from litutils.shared_configs.wandb import WandbConfig


class CallbacksConfig(BaseConfig):
    """Active callbacks configuration"""

    model_checkpoint: bool = False
    progress_bar: bool = True
    early_stopping: bool = False
    batch_size_finder: bool = False
    lr_monitor: bool = True
    optuna_pruning: bool = True

    @model_validator(mode="after")
    def __check_compatible_callbacks(self) -> Self:
        assert not (
            self.optuna_pruning and (self.early_stopping or self.batch_size_finder)
        ), "Optuna pruning is incompatible with early stopping and batch size finder"

        return self


class TrainerConfig(BaseConfig["Trainer"]):
    """[PyTorch Lightning Trainer](https://lightning.ai/docs/pytorch/stable/common/trainer.html) configuration"""

    # Training params
    max_epochs: int = 50
    max_steps: Optional[int] = -1
    early_stopping_patience: int = 5
    log_every_n_steps: int = 8
    fast_dev_run: bool = False
    gradient_clip_val: Optional[float] = 0.5
    accumulate_grad_batches: int = 1
    precision: Union[str, int] = "32-true"
    val_check_interval: Union[int, float] = 1.0

    is_optuna: bool = False

    # Hardware settings
    accelerator: str = "auto"
    devices: Union[int, str] = "auto"
    num_workers: int = Field(default_factory=lambda: torch.get_num_threads())
    matmul_precision: Literal["medium", "high", "highest"] = "medium"

    # Callback settings
    callbacks: CallbacksConfig = Field(default_factory=CallbacksConfig)

    # Logging
    wandb_config: WandbConfig = WandbConfig(
        project="IngredientClassifier",
    )

    # Propagated fields
    is_debug: bool = False
    verbose: bool = True

    def update_wandb_config(self, experiment_config: "ExperimentConfig") -> Self:  # type: ignore
        """Update WandbConfig with experiment specific settings"""
        self.wandb_config.name = experiment_config.run_name
        self.wandb_config.tags = [
            f"model:{str(experiment_config.module_config.model).strip("ModelType.")}",
        ]

        return self

    def setup_target(
        self, experiment_config: "ExperimentConfig", **kwargs: Any  # type: ignore
    ) -> Trainer:
        """Setup trainer using factory pattern"""
        if self.is_debug:
            self._setup_debug_mode()
        return TrainerFactory.create(self, experiment_config, **kwargs)

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

    def __init__(self, config: TrainerConfig, experiment_config: "ExperimentConfig", trial: Optional[Trial]):  # type: ignore
        self.config = config
        self.experiment_config = experiment_config
        self.trial = trial

    @classmethod
    def create(
        cls, config: TrainerConfig, experiment_config: "ExperimentConfig", **kwargs: Any  # type: ignore
    ) -> Trainer:
        """Create and configure a new Trainer instance"""

        factory = cls(config, experiment_config, trial=kwargs.get("trial"))

        # Setup hardware precision
        torch.set_float32_matmul_precision(config.matmul_precision)

        # Assemble components
        callbacks = factory._assemble_callbacks()
        loggers = factory._assemble_loggers()

        # Create trainer
        trainer = Trainer(
            max_epochs=config.max_epochs,
            max_steps=config.max_steps,
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
        )
        return trainer

    def _assemble_callbacks(self) -> List[Callback]:
        """Create and return active callbacks

        Documentation for [PyTorch Lightning Callbacks](https://pytorch-lightning.readthedocs.io/en/stable/extensions/callbacks.html)

        Returns:
            List[Callback]: A list of active callbacks.
        """
        callbacks: List[Callback] = []

        if self.config.callbacks.model_checkpoint:
            dirpath = self.experiment_config.paths.checkpoints / str(
                self.experiment_config.module_config.model
            ).strip("ModelType.")
            dirpath.mkdir(parents=True, exist_ok=True)

            callbacks.append(
                ModelCheckpoint(
                    monitor="val_loss",
                    mode="min",
                    verbose=self.config.verbose,
                    dirpath=dirpath,
                    filename=(
                        self.experiment_config.run_name + "{epoch:02d}-{val_loss:.2f}"
                    ),
                )
            )

        if self.config.callbacks.early_stopping:
            callbacks.append(
                EarlyStopping(
                    monitor="val_loss",
                    patience=self.config.early_stopping_patience,
                    mode="min",
                    verbose=self.config.verbose,
                    check_on_train_epoch_end=True,
                    strict=False,
                )
            )

        if self.config.callbacks.lr_monitor:
            callbacks.append(
                LearningRateMonitor(logging_interval="step", log_momentum=True)
            )

        if self.config.callbacks.progress_bar:
            callbacks.append(CustomTQDMProgressBar())

        if self.config.callbacks.optuna_pruning:
            assert self.trial is not None, "Trial object is required for Optuna pruning"
            callbacks.append(
                self.experiment_config.optuna_config.get_pruning_callback(self.trial)
            )

        return callbacks

    def _assemble_loggers(self) -> Union[List[Logger], bool]:
        """Setup logging systems"""
        if self.config.wandb_config:
            return [self.config.wandb_config.setup_target()]
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
