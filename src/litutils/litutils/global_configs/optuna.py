from typing import Generic, Optional, Sequence, Type, TypeVar, Union, get_args

import optuna
from pydantic import Field

from ..utils import BaseConfig

T = TypeVar("T", int, float, bool, str)


class OptunaOptimizable(BaseConfig, Generic[T]):
    target: Type[T]

    start: Optional[Union[int, float]] = Field(None)
    end: Optional[Union[int, float]] = Field(None)
    step: Optional[int] = Field(1)
    categories: Optional[Sequence[str]] = Field(None)

    def setup_target(self, name: str, trial: optuna.Trial) -> T:

        if self.target is int:
            if self.start is not None and self.end is not None:
                return trial.suggest_int(name, self.start, self.end, step=self.step)  # type: ignore
            else:
                raise ValueError("Integer target requires 'start' and 'end' values.")

        elif self.target is float:
            if self.start is not None and self.end is not None:
                return trial.suggest_float(name, self.start, self.end)  # type: ignore
            else:
                raise ValueError("Float target requires 'start' and 'end' values.")

        elif self.target is bool:
            return trial.suggest_categorical(name, [True, False])  # type: ignore

        elif self.target is str:
            if self.categories is not None:
                return trial.suggest_categorical(name, self.categories)  # type: ignore
            else:
                raise ValueError("Categorical target requires 'categories' values.")

        else:
            raise ValueError(f"Unsupported or misconfigured target type: {self.target}")

    def __str__(self) -> str:
        return (
            f"OptunaOptimizable(start={self.start}, end={self.end}, "
            f"step={self.step}, categories={self.categories}, default={self.default})"
        )
