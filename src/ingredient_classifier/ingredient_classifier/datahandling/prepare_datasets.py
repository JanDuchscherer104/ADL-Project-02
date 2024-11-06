import os
import zipfile
from enum import Enum, auto
from itertools import chain
from pathlib import Path
from random import Random
from subprocess import call, run
from typing import Dict, List, Type
from urllib.request import urlretrieve

from pydantic import Field
from sklearn.model_selection import train_test_split

from litutils import BaseConfig, PathConfig


class DatasetType(Enum):
    FREIBURG_GROCERIES = "freiburg_groceries"
    FRU_VEG_KAGGLE = "fruit-and-vegetable-image-recognition"


class PrepStage(Enum):
    """Stages for dataset preparation"""

    ALL = auto()  # Download and transform
    DOWNLOAD = auto()  # Only download
    TRANSFORM = auto()  # Only transform splits
    NONE = auto()  # No preparation needed


class DatasetPrepConfig(BaseConfig):
    target: Type["DatasetPrep"] = Field(default_factory=lambda: DatasetPrep)
    datasets_tf_stages: Dict[DatasetType, PrepStage] = Field(
        default_factory=lambda: {
            DatasetType.FREIBURG_GROCERIES: PrepStage.NONE,
            DatasetType.FRU_VEG_KAGGLE: PrepStage.TRANSFORM,
        }
    )
    val_to_test_ratio: float = 0.5
    rnd_seed: int = 42
    paths: PathConfig = Field(default_factory=PathConfig)


FREIBURG_URLS = {
    "dataset": "http://aisdatasets.informatik.uni-freiburg.de/freiburg_groceries_dataset/freiburg_groceries_dataset.tar.gz",
    "splits": "http://aisdatasets.informatik.uni-freiburg.de/freiburg_groceries_dataset/splits.tar.gz",
}

KAGGLE_DATASETS = {
    DatasetType.FRU_VEG_KAGGLE: "kritikseth/fruit-and-vegetable-image-recognition"
}


class DatasetPrep:
    def __init__(self, config: DatasetPrepConfig):
        self.config = config

    def prepare_datasets(self):
        """Prepare datasets according to their specified stages"""
        for dataset, stage in self.config.datasets_tf_stages.items():
            match stage:
                case PrepStage.ALL:
                    self.download_and_extract(dataset)
                    self.transform_splits(dataset)
                case PrepStage.DOWNLOAD:
                    self.download_and_extract(dataset)
                case PrepStage.TRANSFORM:
                    self.transform_splits(dataset)
                case PrepStage.NONE:
                    print(f"No preparation needed for {dataset}")

    def download_and_extract(self, dataset_type: DatasetType):
        """Download and extract datasets based on type"""
        match dataset_type:
            case DatasetType.FREIBURG_GROCERIES:
                self._download_and_extract_freiburg_groceries()
            case DatasetType.FRU_VEG_KAGGLE:
                self._download_kaggle_dataset(dataset_type)
            case _:
                print(f"No download needed for {dataset_type}")

    def transform_splits(self, dataset_type: DatasetType):
        """Transform dataset splits based on type"""
        match dataset_type:
            case DatasetType.FREIBURG_GROCERIES:
                self._transform_freiburg_splits()
            case DatasetType.FRU_VEG_KAGGLE:
                self._normalize_vegfru_dir_names()
                self._fix_png_profiles(dataset_type)
                # remove corrupt sample
                if (
                    pth := self.config.paths.data
                    / DatasetType.FRU_VEG_KAGGLE.value
                    / "train/bell_pepper/Image_56.jpg"
                ).exists():
                    pth.unlink()
                    print("Removed corrupt sample")
            case _:
                print(f"No split transformation needed for {dataset_type}")

    def _transform_freiburg_splits(self) -> None:
        """
        Transform Freiburg splits into unified train/val/test splits using functional approach.

        Args:
            val_to_test_ratio: Ratio to split validation from test set
        """
        splits_dir = (
            self.config.paths.data / DatasetType.FREIBURG_GROCERIES.value / "splits"
        )

        if not splits_dir.exists():
            raise FileNotFoundError(f"Splits directory not found: {splits_dir}")

        # Get split files
        train_files = sorted(splits_dir.glob("train*.txt"))
        test_files = sorted(splits_dir.glob("test*.txt"))

        if not (train_files and test_files):
            raise FileNotFoundError("No split files found")

        def read_clean_lines(file: Path) -> list[str]:
            return list(filter(None, map(str.strip, file.read_text().splitlines())))

        # Merge all samples from train files
        train_samples = sorted(
            set(chain.from_iterable(map(read_clean_lines, train_files)))
        )

        # Merge all samples from test files
        test_pool = sorted(set(chain.from_iterable(map(read_clean_lines, train_files))))

        # Split test pool into validation and test using sklearn
        val_samples, test_samples = train_test_split(
            test_pool,
            test_size=1 - self.config.val_to_test_ratio,
            random_state=self.config.rnd_seed,
            shuffle=True,
        )

        # Save splits using Path
        for name, samples in [
            ("train.txt", train_samples),
            ("val.txt", val_samples),
            ("test.txt", test_samples),
        ]:
            (splits_dir / name).write_text("\n".join(samples))

        for file in chain(train_files, test_files):
            file.unlink()
            print(f"Removed old split file: {file.name}")

        # Log statistics with formatting
        print("\nCreated unified splits:")
        print(f"Train: {len(train_samples):>6,d} samples")
        print(f"Val:   {len(val_samples):>6,d} samples")
        print(f"Test:  {len(test_samples):>6,d} samples")

    def _fix_png_profiles(self, dataset_type: DatasetType) -> None:
        """Fix incorrect sRGB profiles in PNG files using ImageMagick's mogrify"""
        # <https://stackoverflow.com/questions/22745076/libpng-warning-iccp-known-incorrect-srgb-profile/22747902#22747902>
        dataset_path = self.config.paths.data / dataset_type.value
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

        print(f"\nFixing PNG profiles in {dataset_path}")

        # Find all directories containing PNGs using pathlib
        png_dirs = {f.parent for f in dataset_path.rglob("*.png")}
        if not png_dirs:
            print("No PNG files found")
            return

        print(f"Found PNG files in {len(png_dirs)} directories")

        # Process each directory
        for dir_path in png_dirs:
            try:
                print(f"Processing {dir_path.relative_to(dataset_path)}...")

                # Get absolute paths for all PNG files in directory
                png_files = list(dir_path.glob("*.png"))
                if png_files:
                    # Convert Path objects to strings for subprocess
                    png_paths = [str(f.absolute()) for f in png_files]
                    run(["mogrify"] + png_paths, check=True, capture_output=True)

            except Exception as e:
                print(f"Error processing directory {dir_path}: {e}")
                continue

        print("✓ PNG profiles fixed")

    def _normalize_vegfru_dir_names(self):
        """Normalize directory names by replacing spaces with underscores in the VeGFru dataset"""
        splits = ["train", "validation", "test"]
        for split in splits:
            dataset_path = (
                self.config.paths.data / DatasetType.FRU_VEG_KAGGLE.value / split
            )
            if not dataset_path.exists():
                raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

            for class_dir in dataset_path.iterdir():
                if class_dir.is_dir():
                    new_name = class_dir.name.replace(" ", "_")
                    new_dir = class_dir.parent / new_name
                    if new_name != class_dir.name and not new_dir.exists():
                        class_dir.rename(new_dir)
                        print(f"Renamed '{class_dir.name}' to '{new_name}'")

        print("Directory names normalized in VeGFru dataset")

    def _download_and_extract_tar_gz(
        self, url: str, dataset_path: Path, filename: str
    ) -> None:
        tar_path = dataset_path / filename

        try:
            print(f"Downloading {filename} to {tar_path}")
            urlretrieve(url, tar_path)

            print(f"Extracting {filename}")
            call(["tar", "-xf", tar_path.as_posix(), "-C", dataset_path.as_posix()])

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            raise
        finally:
            if tar_path.exists():
                tar_path.unlink()
                print(f"Cleaned up {filename}")

    def _download_and_extract_freiburg_groceries(self):
        dataset_path = self.config.paths.data / DatasetType.FREIBURG_GROCERIES.value
        dataset_path.mkdir(parents=True, exist_ok=True)

        # Download and extract main dataset
        self._download_and_extract_file(
            FREIBURG_URLS["dataset"], dataset_path, "dataset.tar.gz"
        )

        # Download and extract splits
        self._download_and_extract_tar_gz(
            FREIBURG_URLS["splits"], dataset_path, "splits.tar.gz"
        )

    def _download_kaggle_dataset(self, dataset_type: DatasetType):
        dataset_path = self.config.paths.data / dataset_type.value
        dataset_path.mkdir(parents=True, exist_ok=True)

        dataset_name = KAGGLE_DATASETS[dataset_type]
        zip_path = dataset_path / f"{dataset_type.value}.zip"

        print(f"Downloading Kaggle dataset {dataset_name}")
        cmd = [
            "kaggle",
            "datasets",
            "download",
            dataset_name,
            "-p",
            str(dataset_path),
        ]
        run(cmd, check=True)

        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(dataset_path)

        print(f"Dataset extracted to: {dataset_path}")

        zip_path.unlink()
        print("Cleaned up zip file")


if __name__ == "__main__":
    config = DatasetPrepConfig()
    dataset_prep = config.setup_target()
    dataset_prep.prepare_datasets()
