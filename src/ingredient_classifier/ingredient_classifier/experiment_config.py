from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple, Type, Union

from pydantic import Field, model_validator
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from typing_extensions import Self

from litutils import CONSOLE, BaseConfig, Optimizable, PathConfig

from .lit_datamodule import DatamoduleParams, LitDataModule
from .lit_module import ImgClassifierParams, LitImageClassifierModule
from .lit_trainer_factory import TrainerConfig


class ExperimentConfig(BaseConfig):
    """
    TODO: add TrainerConfig
    """

    is_debug: bool = True
    verbose: bool = True

    from_ckpt: Optional[str] = None
    is_optuna: bool = False
    is_gpu: bool = True

    is_fast_dev_run: bool = False
    paths: PathConfig = Field(default_factory=PathConfig)
    trainer_config: TrainerConfig = Field(default_factory=TrainerConfig)

    module_config: ImgClassifierParams = ImgClassifierParams()
    module_type: Type[LitImageClassifierModule] = Field(
        default_factory=lambda: LitImageClassifierModule
    )  # Expect a class derived from LightningModule
    datamodule_config: DatamoduleParams = DatamoduleParams()
    datamodule_type: Type[LitDataModule] = Field(
        default_factory=lambda: LitDataModule
    )  # Expect a class derived from LightningDataModule

    def dump_yaml(self) -> None:
        self.to_yaml(self.paths.configs)

    def dump(self):
        i = 1
        config_file = self.paths.configs / f"{self.wandb_config.run_name}.yaml"
        while config_file.exists():
            config_file = self.paths.configs / f"{self.wandb_config.run_name}-{i}.yaml"
            i += 1
        self.to_yaml(config_file)

    @classmethod
    def read(cls, file_name: str, root: Optional[Path] = None) -> Self:
        root = root or Path(__file__).parents[3].resolve()
        config_file = (root / ".configs" / file_name).with_suffix(".yaml")
        assert config_file.exists(), f"{config_file} does not exist"
        return cls.from_yaml(config_file)  # type: ignore

    def setup_target(
        self, **kwargs: Any
    ) -> Tuple[Trainer, LightningModule, LightningDataModule]:
        """Create trainer, module and datamodule instances.

        Returns:
            Tuple containing:
            - PyTorch Lightning Trainer
            - LightningModule instance
            - LightningDataModule instance
        """
        # Setup trainer first
        trainer = self.trainer_config.setup_target(**kwargs)

        # Setup module with checkpoint handling
        if self.from_ckpt:
            try:
                CONSOLE.log(f"Loading model from checkpoint: {self.from_ckpt}")
                lit_module = self.module_type.load_from_checkpoint(
                    checkpoint_path=self.from_ckpt, params=self.module_config
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load checkpoint: {e}")
        else:
            lit_module = self.module_config.setup_target()

        # Setup datamodule
        lit_datamodule = self.datamodule_config.setup_target()

        CONSOLE.log(f"Experiment setup complete!")
        return trainer, lit_module, lit_datamodule
