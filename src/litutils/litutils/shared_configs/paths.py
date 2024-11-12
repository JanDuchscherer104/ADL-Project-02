from abc import ABC, abstractmethod
from pathlib import Path
from typing import Annotated, Type

from pydantic import Field, ValidationInfo, field_validator

from ..utils import SingletonConfig


class PathConfig(SingletonConfig):
    root: Path = Field(default_factory=lambda: Path(__file__).parents[4].resolve())
    target: Type["PathConfig"] = Field(default_factory=lambda: PathConfig)

    data: Annotated[Path, Field(default=".data")]
    webcam_capture: Annotated[Path, Field(default=".data/webcam_capture")]
    checkpoints: Annotated[Path, Field(default=".logs/checkpoints")]
    configs: Annotated[Path, Field(default=".configs")]
    wandb: Annotated[Path, Field(default=".logs/wandb")]
    optuna_study_uri: Annotated[str, Field(default=".logs/optuna/{study_name}.db")]

    # AutoDistill related paths
    autodistill_dataset: Path = Field(
        default_factory=lambda: Path(".data/combined_groceries")
    )
    autodistill_output: Path = Field(
        default_factory=lambda: Path(".data/output/labels")
    )

    @field_validator(
        "data",
        "webcam_capture",
        "checkpoints",
        "configs",
        "wandb",
        "autodistill_dataset",
        "autodistill_output",
        mode="before",
    )
    @classmethod
    def convert_to_path(cls, v: str | Path, info: ValidationInfo) -> Path:
        if isinstance(v, str):
            root = info.data.get("root", Path.cwd())
            v = root / v if not Path(v).is_absolute() else Path(v)
        v = v.resolve()
        v.mkdir(parents=True, exist_ok=True)
        return v

    @field_validator("optuna_study_uri", mode="before")
    @classmethod
    def convert_to_uri(cls, v: str, info: ValidationInfo) -> str:
        study_dir = info.data.get("root", Path.cwd()) / Path(v).parent
        study_dir.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{(study_dir / Path(v).name).as_posix()}"
