from typing import Dict, List, Optional, Tuple, Type

import cv2
from pydantic import Field
from torch import Tensor
from torch.utils.data import Dataset

from litutils import BaseConfig, PathConfig, Stage

from .transforms import Transforms


class FreiburgGroceriesDatasetParams(BaseConfig["FreiburgGroceriesDataset"]):
    """Parameters for FreiburgGroceriesDataset"""

    paths: PathConfig = Field(default_factory=PathConfig)
    stage: Stage = Field(
        default=Stage.TRAIN, description="Dataset stage (TRAIN/VAL/TEST)"
    )

    target: Type["FreiburgGroceriesDataset"] = Field(
        default_factory=lambda: FreiburgGroceriesDataset
    )


class FreiburgGroceriesDataset(Dataset):
    """PyTorch Dataset for Freiburg Groceries with unified splits"""

    def __init__(
        self,
        params: FreiburgGroceriesDatasetParams,
        transforms: Optional[Transforms] = None,
    ):
        """Initialize dataset with specified parameters."""
        self.params = params
        self.data_dir = params.paths.data / "freiburg_groceries"
        assert self.data_dir.exists(), f"Data directory not found: {self.data_dir}"

        # transforms or identiy function
        self.transforms = transforms or (lambda *x: x)

        # Map stage to split file name
        split_map = {
            Stage.TRAIN: "train.txt",
            Stage.VAL: "val.txt",
            Stage.TEST: "test.txt",
        }

        # Load split file
        split_file = self.data_dir / "splits" / split_map[params.stage]
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")

        def process_line(line):
            parts = line.strip().split()
            if parts:  # Skip empty lines
                path, label = parts  # Unpack into label and remaining path parts
                return (path, int(label))
            return None

        # Read and clean image paths
        with split_file.open() as f:
            self.samples: List[Tuple[str, int]] = list(
                filter(None, map(process_line, f.readlines()))
            )

        if not self.samples:
            raise ValueError(f"No samples found in {split_file}")

        # Create class index mapping
        self.class_to_idx: Dict[str, int] = dict(
            map(lambda t: (t[0].split("/")[0], int(t[1])), self.samples)
        )
        self.idx_to_class: Dict[int, str] = dict(
            map(reversed, self.class_to_idx.items())  # type: ignore
        )

        # Sort based on the class index
        self.classes = [
            class_name
            for class_name, _ in sorted(self.class_to_idx.items(), key=lambda x: x[1])
        ]

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """Get image and label for given index."""
        img_path, label = self.samples[idx]
        img_path = self.data_dir / "images" / img_path

        # Load image
        image = cv2.imread(img_path.as_posix())  # type: ignore
        if image is None:
            raise RuntimeError(f"Failed to load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return self.transforms(image, label)  # type: ignore

    @property
    def num_classes(self) -> int:
        """Return number of classes in dataset."""
        return len(self.classes)
