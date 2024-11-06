import json
from pathlib import Path
from typing import Annotated, Dict, List, Tuple, Type, Union

import pandas as pd
import torch
from pydantic import Field, ValidationInfo, field_validator
from torch import Tensor
from torch.utils.data import Dataset

from litutils import CONSOLE, BaseConfig, PathConfig, Stage

from .ds_freiburg_groceries import FreiburgGroceriesDatasetParams
from .ds_fruit_veg import FruitVegDatasetParams
from .transforms import TransformsConfig, TransformsType


class CompositeDatasetParams(BaseConfig["CompositeGroceryDataset"]):
    """Parameters for CompositeDataset"""

    paths: PathConfig = Field(default_factory=PathConfig)

    stage: Stage = Field(default=Stage.TRAIN)

    from_metadata: bool = False
    """Load dataset from saved metadata instead of building mappings"""
    save_metadata: bool = True
    """Wheter to save metadata after building mappings"""

    transforms_type: TransformsType = TransformsType.TRAIN_FROM_SCRATCH
    transforms_config: Annotated[TransformsConfig, Field(None)]

    dataset_params: List[
        Union[FreiburgGroceriesDatasetParams, FruitVegDatasetParams]
    ] = Field(default_factory=list, description="List of dataset parameters")

    validate_counts: bool = False
    """Validate that sample counts match source datasets after filtering"""

    classes_to_drop: List[str] = Field(
        default_factory=lambda: [
            "candy",
            "cereal",
            "cake",
            "chips",
            "chocolate",
            "coffee",
            "soda",
            "tea",
            "water",
            "jam",
            "juice",
        ],
        description="Classes to exclude - processed foods and beverages",
    )

    target: Type["CompositeGroceryDataset"] = Field(
        default_factory=lambda: CompositeGroceryDataset
    )

    @field_validator("transforms_config", mode="before")
    @classmethod
    def __init_transforms_config(cls, _, info: ValidationInfo) -> TransformsConfig:
        tf_type = info.data["transforms_type"]
        match info.data["stage"]:
            case Stage.TRAIN:
                assert tf_type in {
                    TransformsType.TRAIN_FROM_SCRATCH,
                    TransformsType.TRAIN_FINE_TUNE,
                }
                CONSOLE.log(
                    f"Using transforms type: {tf_type} for split: {info.data['stage']}"
                )
                return TransformsConfig(transform_type=tf_type)
            case Stage.VAL | Stage.TEST:
                return TransformsConfig(stage=TransformsType.VAL)

    @field_validator("dataset_params", mode="before")
    @classmethod
    def init_dataset_params(
        cls,
        v: List[Union[FreiburgGroceriesDatasetParams, FruitVegDatasetParams]],
        info: ValidationInfo,
    ) -> List[Union[FreiburgGroceriesDatasetParams, FruitVegDatasetParams]]:
        """Initialize dataset parameters if not provided"""
        if v:
            return v

        # Create default params
        return [
            FreiburgGroceriesDatasetParams(
                stage=info.data["stage"], paths=info.data["paths"]
            ),
            FruitVegDatasetParams(stage=info.data["stage"], paths=info.data["paths"]),
        ]

    @property
    def metadata_dir(self) -> Path:
        metadata_dir = self.paths.data / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        return metadata_dir  # type: ignore

    @property
    def class_file(self) -> Path:
        return self.metadata_dir / f"{self.stage.value[0]}_class_to_idx.json"

    @property
    def samples_file(self) -> Path:
        return self.metadata_dir / f"{self.stage.value[0]}_samples.csv"


class CompositeGroceryDataset(Dataset):
    """Efficiently combines and filters multiple grocery datasets with consistent class mapping"""

    CLASS_MAPPING = {
        "corn": "corn",
        "sweetcorn": "corn",
        "beans": "beans",
        "soy_beans": "beans",
        "tomato": "tomato",
        "tomato_sauce": "tomato",
    }

    def __init__(self, params: CompositeDatasetParams) -> None:
        self.params = params
        self.transforms = params.transforms_config.setup_target()

        self.datasets = tuple(
            param.setup_target(transforms=None) for param in params.dataset_params
        )

        # Init mappings and samples
        if params.from_metadata:
            CONSOLE.log(f"Loading {params.stage} dataset from metadata...")
            CONSOLE.warn(
                f"If changes were made to CompositeGroceryDataset.datasets, set from_metadata=False"
            )

            self.classes, self.class_to_idx, self.idx_to_class, self.samples = (
                self._load_from_metadata()
            )
        else:
            CONSOLE.log(f"Building dataset mappings for split {params.stage}...")
            self.classes, self.class_to_idx, self.idx_to_class, self.samples = (
                self._build_mappings()
            )
            # todo: add lengths of composite datasets
            CONSOLE.log(
                f"Found {len(self.samples)} valid samples, coming from {len(self.datasets)} datasets"
            )
            for idx, dataset in enumerate(self.datasets):
                CONSOLE.log(
                    f"Dataset {dataset.__class__.__name__}(idx={idx}, split={params.stage}) has {len(dataset)} samples"
                )

            if self.params.save_metadata:
                self.save_metadata()

        if params.validate_counts:
            self._validate_sample_counts()

        if not self.samples:
            raise ValueError("No valid samples after filtering")

    def _validate_sample_counts(self) -> None:
        """Validate sample counts per dataset and class"""
        import pandas as pd
        from rich.table import Table

        # Initialize counts dictionary
        counts = {
            "Freiburg": {},
            "FruitVeg": {},
        }  # type: Dict[str, Dict[str, int]]

        # Count samples per dataset and class
        for dataset_idx, dataset in enumerate(self.datasets):
            source = "Freiburg" if dataset_idx == 0 else "FruitVeg"

            # Build class mapping for this dataset
            class_mapping = {}
            for idx, cls in enumerate(dataset.classes):
                norm_name = cls.replace(" ", "_").lower()
                mapped = self.CLASS_MAPPING.get(norm_name, norm_name)
                if mapped in self.class_to_idx:
                    class_mapping[idx] = mapped
                    counts[source][mapped] = 0

            # Count valid samples per class
            for _, label in dataset:
                label_val = label.item() if isinstance(label, torch.Tensor) else label
                if label_val in class_mapping:
                    counts[source][class_mapping[label_val]] += 1

        # Create DataFrame
        df = pd.DataFrame(counts).fillna(0)
        df["Total"] = df.sum(axis=1)
        df.loc["Total"] = df.sum()

        # Create rich table
        table = Table(title="Sample Distribution")
        table.add_column("Class", justify="left", style="cyan")
        table.add_column("Freiburg", justify="right", style="green")
        table.add_column("FruitVeg", justify="right", style="green")
        table.add_column("Total", justify="right", style="bold white")

        # Add rows
        for cls in df.index[:-1]:  # Exclude total row
            table.add_row(
                cls,
                f"{int(df.loc[cls, 'Freiburg']):,}",
                f"{int(df.loc[cls, 'FruitVeg']):,}",
                f"{int(df.loc[cls, 'Total']):,}",
            )

        # Add total row
        table.add_row(
            "Total",
            f"{int(df.loc['Total', 'Freiburg']):,}",
            f"{int(df.loc['Total', 'FruitVeg']):,}",
            f"{int(df.loc['Total', 'Total']):,}",
            style="bold",
        )

        # Print table
        CONSOLE.print(table)

        # Validate total counts match samples
        actual_counts = [
            sum(1 for x in self.samples if x[0] == idx)
            for idx in range(len(self.datasets))
        ]

        expected_counts = [
            int(df.loc["Total", "Freiburg"]),
            int(df.loc["Total", "FruitVeg"]),
        ]

        if expected_counts != actual_counts:
            CONSOLE.warn(
                "Sample count mismatch:\n"
                f"Expected counts: {expected_counts}\n"
                f"Actual counts: {actual_counts}"
            )
            raise ValueError(
                "Sample counts don't match source datasets after filtering"
            )

        CONSOLE.log("✓ Sample counts validated successfully")

    def _build_mappings(
        self,
    ) -> Tuple[List[str], Dict[str, int], Dict[int, str], List[Tuple[int, int, int]]]:
        # Collect valid classes and build sample mapping
        valid_classes = set()
        valid_samples = []

        # Pre-compile set of classes to drop
        drop_classes = set(self.params.classes_to_drop)

        # First pass - collect valid classes
        for dataset_idx, dataset in enumerate(self.datasets):
            for class_name in dataset.classes:
                norm_name = class_name.replace(" ", "_").lower()
                if norm_name not in drop_classes:
                    mapped = self.CLASS_MAPPING.get(norm_name, norm_name)
                    if mapped not in drop_classes:
                        valid_classes.add(mapped)

        # Create sorted class list and mapping
        classes: List[str] = sorted(valid_classes)
        class_to_idx: Dict[str, int] = {cls: idx for idx, cls in enumerate(classes)}
        idx_to_class: Dict[int, str] = dict(map(reversed, class_to_idx.items()))  # type: ignore

        # Second pass - collect valid samples
        offset = 0
        skipped_samples = 0
        for dataset_idx, dataset in enumerate(self.datasets):
            class_mapping = {}
            for idx, cls in enumerate(dataset.classes):
                norm_name = cls.replace(" ", "_").lower()
                mapped = self.CLASS_MAPPING.get(norm_name, norm_name)
                if mapped in class_to_idx:
                    class_mapping[idx] = class_to_idx[mapped]

            # Collect valid samples
            for idx in range(len(dataset)):
                try:
                    _, label = dataset[idx]
                    label_val = (
                        label.item() if isinstance(label, torch.Tensor) else label
                    )
                    if label_val in class_mapping:
                        valid_samples.append(
                            (dataset_idx, idx, class_mapping[label_val])
                        )
                except Exception as e:
                    skipped_samples += 1
                    CONSOLE.warn(f"Error processing sample {idx + offset}: {str(e)}")

            offset += len(dataset)

        if skipped_samples:
            CONSOLE.warn(f"Skipped {skipped_samples} invalid samples during processing")

        return classes, class_to_idx, idx_to_class, valid_samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        dataset_idx, sample_idx, target_class = self.samples[idx]
        img, _ = self.datasets[dataset_idx][sample_idx]
        return self.transforms(X=img, y=target_class)  # type: ignore

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def _load_from_metadata(
        self,
    ) -> Tuple[List[str], Dict[str, int], Dict[int, str], List[Tuple[int, int, int]]]:
        """Load dataset state from metadata"""
        # Load class mapping
        with (self.params.class_file).open() as f:
            class_to_idx = json.load(f)

        # Derive other mappings
        classes = [
            class_name
            for class_name, _ in sorted(class_to_idx.items(), key=lambda x: x[1])
        ]
        idx_to_class = dict(map(reversed, class_to_idx.items()))  # type: ignore

        # Load samples
        samples_df = pd.read_csv(self.params.samples_file)
        samples = list(
            map(tuple, samples_df[["dataset_idx", "sample_idx", "target_class"]].values)
        )

        CONSOLE.log(f"Loaded {len(samples)} samples from {self.params.samples_file}")

        return classes, class_to_idx, idx_to_class, samples

    def save_metadata(self) -> None:
        """Save dataset metadata to configured directory"""

        # Save class mapping
        with (self.params.class_file).open("w") as f:
            json.dump(self.class_to_idx, f, indent=2)
        CONSOLE.log(f"Saved class mappings to {self.params.class_file}")

        # Save samples with human readable info
        samples_df = pd.DataFrame(
            self.samples, columns=["dataset_idx", "sample_idx", "target_class"]
        )
        samples_df["class"] = samples_df["target_class"].map(self.idx_to_class)
        samples_df["source"] = samples_df["dataset_idx"].map(
            {0: "Freiburg", 1: "FruitVeg"}
        )
        samples_df.to_csv(self.params.samples_file, index=False)

        CONSOLE.log(f"Saved {len(samples_df)} samples to {self.params.samples_file}")
