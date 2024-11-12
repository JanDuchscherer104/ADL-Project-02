from pathlib import Path
from typing import Any, Literal, Optional, Type, Union

import wandb
import wandb.wandb_run
from pydantic import Field
from pytorch_lightning.loggers import WandbLogger

from ..utils import BaseConfig
from .paths import PathConfig

# from wandb.sdk.lib import RunDisabled
# from wandb.wandb_run import Run


class WandbConfig(BaseConfig):
    # Essential fields directly matching WandbLogger parameters
    name: Optional[str] = Field(None, description="Display name for the run.")
    project: Optional[str] = Field(..., description="Name of the wandb project.")
    save_dir: Path = Field(
        default_factory=lambda: PathConfig().wandb,
        serialization_alias="dir",
        description="Directory to save wandb logs.",
    )
    version: Optional[str] = Field(
        None,
        description="Version ID for resuming runs.",
    )
    offline: Optional[bool] = Field(
        False,
        description="Run offline mode.",
    )
    log_model: Union[Literal["all"], bool] = Field(
        False,
        description="Log model checkpoints as wandb artifacts.",
    )
    prefix: Optional[str] = Field(
        "",
        description="Str prefix for beginning of metric keys.",
    )
    # experiment: Optional[Union[Run, RunDisabled]] = Field(
    #     None, description="Predefined wandb run object."
    # )
    checkpoint_name: Optional[str] = Field(
        None, description="Name of model checkpoint artifact."
    )
    tags: Optional[list[str]] = Field(
        None, description="List of tags for easier filtering."
    )

    group: Optional[str] = Field(None, description="Group name for multiple runs.")
    job_type: Optional[str] = Field(None, description="Type of job for the run.")
    target: Type[WandbLogger] = Field(None, exclude=True, validate_default=False)

    def setup_target(self, **kwargs: Any) -> WandbLogger:
        """
        Initializes the [WandbLogger](https://lightning.ai/docs/pytorch/stable/extensions/generated/lightning.pytorch.loggers.WandbLogger.html) with the specified configurations.

        For kwargs refer to [wandb.init](https://docs.wandb.ai/ref/python/init/)
        """
        logger = WandbLogger(
            name=self.name,
            save_dir=self.save_dir,
            version=self.version,
            offline=self.offline,
            project=self.project,
            log_model=self.log_model,
            prefix=self.prefix,
            experiment=wandb.run,
            checkpoint_name=self.checkpoint_name,
            tags=self.tags,
            group=self.group,
            job_type=self.job_type,
            **(kwargs or {}),
        )

        return logger
