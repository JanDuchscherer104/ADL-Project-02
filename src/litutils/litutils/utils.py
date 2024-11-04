import traceback
from enum import Enum
from functools import wraps
from pathlib import Path
from threading import Lock
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Generic,
    Optional,
    Type,
    TypeVar,
    Union,
)

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_yaml import parse_yaml_file_as, to_yaml_file
from rich.console import Console as RichConsole


class _Console(RichConsole):
    _instance: RichConsole = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(_Console, cls).__new__(cls)
            cls._instance.__init__(*args, **kwargs)
        return cls._instance

    def warn(self, message: str) -> None:
        # Capture the full stack trace
        stack_trace = traceback.format_stack(limit=3)[:-1]
        stack_trace_formatted = "".join(stack_trace)

        # Print the warning message with the stack trace
        self.print(
            f"[bright_yellow]Warning:[/bright_yellow] {message}\n"
            f"[dim]{stack_trace_formatted}[/dim]"
        )


CONSOLE = _Console(width=120)


TargetType = TypeVar("TargetType")


class NoTarget:
    @staticmethod
    def setup_target(config: "BaseConfig", **kwargs: Any) -> None:
        CONSOLE.warn(
            f"No target specified for config '{config.__class__.__name__}'. Returning None!"
        )
        return None


class BaseConfig(BaseModel, Generic[TargetType]):
    target: Callable[["BaseConfig[TargetType]"], TargetType] = Field(
        default_factory=lambda: NoTarget
    )

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_default=True)

    @classmethod
    def from_yaml(
        cls: Type["BaseConfig[TargetType]"], file: Union[Path, str]
    ) -> "BaseConfig[TargetType]":
        return cls.model_validate(parse_yaml_file_as(cls, file))  # type: ignore

    def to_yaml(self, file: Union[Path, str]) -> None:
        to_yaml_file(file, self, indent=4)

    def __str__(self) -> str:
        lines = [self.__class__.__name__ + ":"]
        for key, val in self.model_dump().items():
            if isinstance(val, tuple):
                val = "[" + ", ".join(map(str, val)) + "]"
            lines.append(f"{key}: {val}")
        return "\n    ".join(lines)

    def setup_target(self, **kwargs: Any) -> TargetType:
        if not callable(factory := getattr(self.target, "setup_target", self.target)):
            CONSOLE.print(
                f"Target '[bold yellow]{self.target}[/bold yellow]' of type [bold yellow]{factory.__class__.__name__}[/bold yellow] is not callable."
            )
            raise ValueError(
                f"Target '{self.target}' of type {factory.__class__.__name__} is not callable / does not have a 'setup_target' or '__init__' method."
            )

        return factory(self, **kwargs)

    def inspect(self) -> str:
        lines = [self.__class__.__name__ + ":"]
        for field_name, field in self.model_fields.items():
            lines.append(
                f'{field_name}: (value={getattr(self, field_name)}, type={field.annotation.__name__}, description="{field.description}")'
            )
        return "\n    ".join(lines)


class SingletonConfig(BaseConfig):
    """Base class for singleton configurations."""

    _instances: ClassVar[Dict[Type, Any]] = {}
    _lock: ClassVar[Lock] = Lock()

    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True, validate_default=True
    )

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                # Create new instance via Pydantic's BaseModel.__new__
                instance = super(BaseConfig, cls).__new__(cls)
                # Set initialization flag in dict to avoid Pydantic validation
                instance.__dict__["_initialized"] = False
                cls._instances[cls] = instance
            return cls._instances[cls]

    def __init__(self, **kwargs):
        if not getattr(self, "_initialized", False):
            # Only initialize once via Pydantic
            super().__init__(**kwargs)
            self.__dict__["_initialized"] = True
        else:
            # Update existing instance if new values provided
            for key, value in kwargs.items():
                if hasattr(self, key):
                    current = getattr(self, key)
                    if current != value:
                        CONSOLE.warn(
                            f"Updating singleton {self.__class__.__name__} "
                            f"field '{key}' from {current} to {value}"
                        )
                    setattr(self, key, value)

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self


class Stage(Enum):
    TRAIN = ("fit", "train")
    VAL = ("validate", "val")
    TEST = ("test",)

    def __init__(self, *values):
        self.values = values

    def __str__(self):
        return self.values[0]

    @classmethod
    def from_str(cls, value: Optional[str]) -> Optional["Stage"]:
        for member in cls:
            if value in member.values:
                return member
        return None
