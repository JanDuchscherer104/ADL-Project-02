from enum import Enum
from subprocess import call
from typing import List, Type
from urllib.request import urlretrieve

from pydantic import Field

from litutils import BaseConfig, PathConfig


class DatasetType(Enum):
    FREIBURG_GROCERIES = "freiburg_groceries"


class DatasetPrepConfig(BaseConfig):
    target: Type["DatasetPrep"] = Field(default_factory=lambda: DatasetPrep)
    datasets: List[DatasetType] = Field(
        default_factory=lambda: [DatasetType.FREIBURG_GROCERIES]
    )
    paths: PathConfig = Field(default_factory=PathConfig)


class DatasetPrep:
    def __init__(self, config: DatasetPrepConfig):
        self.config = config

    def download_and_extract(self, dataset_type: DatasetType):
        match dataset_type:
            case DatasetType.FREIBURG_GROCERIES:
                self._download_and_extract_freiburg_groceries()
            case _:
                raise ValueError(f"Dataset URL for {dataset_type} not found.")

    def _download_and_extract_freiburg_groceries(self):
        dataset_url = "http://aisdatasets.informatik.uni-freiburg.de/freiburg_groceries_dataset/freiburg_groceries_dataset.tar.gz"
        dataset_path = self.config.paths.data / DatasetType.FREIBURG_GROCERIES.value
        tar_path = dataset_path / "dataset.tar.gz"

        # Ensure directory exists
        dataset_path.mkdir(parents=True, exist_ok=True)

        try:
            print(f"Downloading freiburg_groceries dataset to {tar_path}")
            urlretrieve(dataset_url, tar_path)

            print("Extracting freiburg_groceries dataset")
            call(["tar", "-xf", tar_path.as_posix(), "-C", dataset_path.as_posix()])

        except Exception as e:
            print(f"Error processing dataset: {e}")
            raise
        finally:
            if tar_path.exists():
                tar_path.unlink()
                print("Cleaned up temporary files")

    def prepare_datasets(self):
        for dataset in self.config.datasets:
            self.download_and_extract(dataset)


if __name__ == "__main__":
    config = DatasetPrepConfig()
    dataset_prep = config.setup_target()
    dataset_prep.prepare_datasets()
