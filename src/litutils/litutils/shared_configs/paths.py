from pathlib import Path
from typing import Annotated, Type

from pydantic import Field, ValidationInfo, field_validator, model_validator

from ..utils import SingletonConfig


class PathConfig(SingletonConfig):
    root: Path = Field(default_factory=lambda: Path(__file__).parents[4].resolve())
    target: Type["PathConfig"] = Field(default_factory=lambda: PathConfig)

    data: Annotated[Path, Field(default=".data")]
    webcam_capture: Annotated[Path, Field(default=".data/webcam_capture")]
    checkpoints: Annotated[Path, Field(default=".logs/checkpoints")]
    tb_logs: Annotated[Path, Field(default=".logs/tb_logs")]
    configs: Annotated[Path, Field(default=".configs")]
    wandb: Annotated[Path, Field(default=".logs/wandb")]

    # @field_validator(
    #     "data",
    #     "webcam_capture",
    #     "checkpoints",
    #     "tb_logs",
    #     "configs",
    #     "wandb",
    #     mode="before",
    # )
    # @classmethod
    # def __convert_to_path(cls, v: str, info: ValidationInfo) -> Path:
    #     root = info.data.get("root")
    #     path = (root / v).resolve() if not Path(v).is_absolute() else Path(v)
    #     assert isinstance(path, Path)
    #     path.mkdir(parents=True, exist_ok=True)
    #     return path
    @field_validator(
        "data",
        "webcam_capture",
        "checkpoints",
        "tb_logs",
        "configs",
        "wandb",
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

    # @field_validator("mlflow_uri", mode="before")
    # @classmethod
    # def __convert_to_uri(cls, v: str, info: ValidationInfo) -> str:
    #     root = info.data.get("root")
    #     if v.startswith("file://"):
    #         return v
    #     uri_path = root / v if not Path(v).is_absolute() else Path(v)
    #     assert isinstance(uri_path, Path)
    #     uri_path.parent.mkdir(parents=True, exist_ok=True)
    #     if not uri_path.exists():
    #         uri_path.mkdir(parents=True, exist_ok=True)
    #     return uri_path.resolve().as_uri()
