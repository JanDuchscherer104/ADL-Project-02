from pathlib import Path
from typing import Annotated, Type

from pydantic import Field, ValidationInfo, field_validator

from ..utils import BaseConfig

ROOT = Path(__file__).parents[3].resolve()


class PathConfig(BaseConfig):
    target: Type["PathConfig"] = Field(default_factory=lambda: PathConfig)

    root: Path = Field(default=ROOT)
    data: Annotated[Path, Field(default=".data")]
    checkpoints: Annotated[Path, Field(default=".logs/checkpoints")]
    tb_logs: Annotated[Path, Field(default=".logs/tb_logs")]
    configs: Annotated[Path, Field(default=".configs")]
    mlflow_uri: Annotated[str, Field(default=".logs/mlflow_logs/mlflow")]

    @field_validator("data", "checkpoints", "tb_logs", "configs")
    @classmethod
    def __convert_to_path(cls, v: str, info: ValidationInfo) -> Path:
        root: Path = info.data.get("root")
        path = (root / v).resolve() if not Path(v).is_absolute() else Path(v)
        path.mkdir(parents=True, exist_ok=True)

        return path

    @field_validator("mlflow_uri")
    @classmethod
    def __convert_to_uri(cls, v: str, info: ValidationInfo) -> str:
        if v.startswith("file://"):
            return v
        root: Path = info.data.get("root")
        uri_path = root / v if not Path(v).is_absolute() else Path(v)
        uri_path.parent.mkdir(parents=True, exist_ok=True)
        if not uri_path.exists():
            uri_path.mkdir(parents=True, exist_ok=True)
        return uri_path.resolve().as_uri()
