import traceback
from enum import Enum
from functools import wraps
from pathlib import Path
from threading import Lock
from typing import Any, Callable, ClassVar, Dict, Generic, Optional, Type, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator
from rich.console import Console as RichConsole
from rich.syntax import Syntax
from rich.text import Text
from rich.theme import Theme
from rich.tree import Tree


class _Console(RichConsole):
    _instance = None
    _lock: Lock = Lock()

    def __new__(cls, *args, verbose: bool = True, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(_Console, cls).__new__(cls)
                cls._instance.verbose = verbose
                cls._instance.__init__(*args, **kwargs)
            return cls._instance

    def __init__(self, *args, **kwargs):
        if not hasattr(self, "_initialized"):
            kwargs["force_terminal"] = True
            kwargs["color_system"] = "auto"
            kwargs["markup"] = True  # Ensure markup is enabled
            kwargs["highlight"] = True  # Enable syntax highlighting
            super().__init__(*args, **kwargs)
            self._initialized = True

    def _get_caller_stack(self) -> str:
        """Get formatted stack trace excluding Console internals"""
        stack = traceback.extract_stack()
        # Filter out frames from this file
        current_file = Path(__file__).resolve()
        relevant_frames = [
            frame
            for frame in stack[:-1]  # Exclude current frame
            if Path(frame.filename).resolve() != current_file
        ]
        # Format remaining frames
        return "".join(
            traceback.format_list(relevant_frames[-2:])
        )  # Show last 2 relevant frames

    def warn(self, message: str) -> None:
        if self.verbose:
            stack_trace = self._get_caller_stack()
            self.print(
                f"[bright_yellow]Warning:[/bright_yellow] {message}\n"
                f"[dim]{stack_trace}[/dim]"
            )

    def log(self, message: str) -> None:
        if self.verbose:
            self.print(message)

    def set_verbose(self, verbose: bool) -> None:
        self.verbose = verbose


CONFIG_THEME = Theme(
    {
        "config.name": "bold blue",  # Config class names
        "config.field": "green",  # Regular fields
        "config.propagated": "yellow",  # Propagated fields
        "config.value": "white",  # Field values
        "config.type": "dim",  # Type annotations
        "config.doc": "italic dim",  # Documentation  # Changed from 'grey' to 'gray'
    }
)

CONSOLE = _Console(
    width=120,
    verbose=True,
    force_terminal=True,
    color_system="auto",
    markup=True,
    highlight=True,
    theme=CONFIG_THEME,
)


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

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_default=True,
        validate_assignment=True,
        protected_namespaces=(),
    )

    propagated_fields: Dict[str, Any] = Field(
        default_factory=dict,
        exclude=True,
        description="Tracks fields propagated from parent configs",
    )

    def setup_target(self, **kwargs: Any) -> TargetType:
        if not callable(factory := getattr(self.target, "setup_target", self.target)):
            CONSOLE.print(
                f"Target '[bold yellow]{self.target}[/bold yellow]' of type [bold yellow]{factory.__class__.__name__}[/bold yellow] is not callable."
            )
            raise ValueError(
                f"Target '{self.target}' of type {factory.__class__.__name__} is not callable / does not have a 'setup_target' or '__init__' method."
            )

        return factory(self, **kwargs)

    def inspect(self, show_docs: bool = False) -> None:
        """Pretty print config structure using rich.

        Args:
            indent: Base indentation level
            show_docs: Whether to show field descriptions and docstrings
        """
        tree = self._build_tree(show_docs=show_docs)
        CONSOLE.print(tree, soft_wrap=False, highlight=True, markup=True, emoji=False)

    def _build_tree(self, show_docs: bool = False) -> Tree:
        """Build rich Tree representation of config."""
        # Create root node with themed style
        tree = Tree(Text(self.__class__.__name__, style="config.name"))

        # Add class docstring if requested
        if show_docs and self.__class__.__doc__:
            tree.add(Text(self.__class__.__doc__, style="config.doc"))

        for field_name, field in self.model_fields.items():
            value = getattr(self, field_name)

            # Style for propagated vs regular fields
            field_style = (
                "config.propagated"
                if field_name in self.propagated_fields
                else "config.field"
            )

            # Create field node
            field_text = Text()
            field_text.append(f"{field_name}: ", style=field_style)

            # Handle nested BaseConfig
            if isinstance(value, BaseConfig):
                subtree = tree.add(field_text)
                nested_tree = value._build_tree(show_docs=show_docs)
                subtree.add(nested_tree)
                continue

            # Handle list/tuple of BaseConfigs
            if (
                isinstance(value, (list, tuple))
                and value
                and isinstance(value[0], BaseConfig)
            ):
                subtree = tree.add(field_text)
                for i, item in enumerate(value):
                    item_tree = item._build_tree(show_docs=show_docs)
                    subtree.add(Text(f"[{i}]", style="config.field")).add(item_tree)
                continue

            # Format value with proper styling
            if isinstance(value, str):
                value_str = f'"{value}"'
            elif isinstance(value, (int, float, bool, Enum)):
                value_str = str(value)
            else:
                value_str = f"<{value.__class__.__name__}>"
            field_text.append(value_str, style="config.value")

            # Add type annotation
            type_name = getattr(field.annotation, "__name__", str(field.annotation))
            field_text.append(f" ({type_name})", style="config.type")

            # Add field node
            field_node = tree.add(field_text)

            # Add documentation if requested
            if show_docs and field.description:
                field_node.add(Text(field.description, style="config.doc"))

        return tree

    @model_validator(mode="after")
    def _propagate_shared_fields(self) -> "BaseConfig":
        """Propagate shared fields to nested BaseConfig instances"""
        for field_name, field_value in self:

            # If field is another BaseConfig
            if isinstance(field_value, BaseConfig):
                self._propagate_to_child(field_name, field_value)

            # Handle lists/tuples of BaseConfigs
            elif isinstance(field_value, (list, tuple)):
                for item in field_value:
                    if isinstance(item, BaseConfig):
                        self._propagate_to_child(field_name, item)

        return self

    def _propagate_to_child(
        self, parent_field: str, child_config: "BaseConfig"
    ) -> None:
        """Propagate matching fields from parent to child config"""
        shared_fields = {
            name: value
            for name, value in self
            if name in child_config.model_fields
            and name != parent_field
            and name not in ("propagated_fields", "target")
        }

        for name, value in shared_fields.items():
            current_value = getattr(child_config, name, None)
            if current_value != value:
                setattr(child_config, name, value)
                child_config.propagated_fields[name] = value

                CONSOLE.log(
                    f"Propagated {name}={value} from "
                    f"{self.__class__.__name__} to {child_config.__class__.__name__}"
                )


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
    TRAIN = ("train", "fit")
    VAL = ("val", "validate")
    TEST = ("test",)

    def __str__(self):
        return self.value[0]

    @classmethod
    def from_str(cls, value: Optional[str]) -> Optional["Stage"]:
        for member in cls:
            if value in member.value:
                return member
        return None
