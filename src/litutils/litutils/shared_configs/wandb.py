from typing import Any, Optional, Type, Union

from pydantic import Field
from pytorch_lightning.loggers import WandbLogger

from ..utils import BaseConfig


class WandbConfig(BaseConfig):
    # Essential fields directly matching WandbLogger parameters
    name: Optional[str] = Field(None, description="Display name for the run.")
    save_dir: Optional[str] = Field(None, description="Directory to save wandb logs.")
    version: Optional[str] = Field(None, description="Version ID for resuming runs.")
    offline: Optional[bool] = Field(False, description="Run offline mode.")
    dir: Optional[str] = Field(None, description="Alternative to save_dir.")
    id: Optional[str] = Field(
        None, description="Same as version; unique ID for the run."
    )
    anonymous: Optional[bool] = Field(False, description="Enable anonymous logging.")
    project: Optional[str] = Field(..., description="Name of the wandb project.")
    log_model: Optional[Union[str, bool]] = Field(
        False, description="Log model checkpoints as wandb artifacts."
    )
    prefix: Optional[str] = Field("", description="Prefix for metric keys.")
    experiment: Optional[Any] = Field(
        None, description="Predefined wandb experiment object."
    )
    checkpoint_name: Optional[str] = Field(
        None, description="Name of model checkpoint artifact."
    )

    target: Type[WandbLogger] = Field(None, validate_default=False)

    def setup_target(self, **kwargs: Any) -> WandbLogger:
        """
        Initializes the WandbLogger with the specified configurations.
        """
        logger = WandbLogger(
            name=self.name,
            save_dir=self.save_dir,
            version=self.version,
            offline=self.offline,
            dir=self.dir,
            id=self.id,
            anonymous=self.anonymous,
            project=self.project,
            log_model=self.log_model,
            prefix=self.prefix,
            experiment=self.experiment,
            checkpoint_name=self.checkpoint_name,
            **(kwargs or {}),
        )

        return logger
