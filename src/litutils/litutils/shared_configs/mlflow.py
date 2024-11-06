import re
from typing import Annotated

import mlflow
from pydantic import Field

from ..utils import CONSOLE, BaseConfig


class MLflowConfig(BaseConfig):
    experiment_name: str = "nosmod"
    run_name: Annotated[str, Field(default="None")]
    experiment_id: Annotated[str, Field(default="None")]

    def setup_mlflow(self, mlflow_uri: str) -> None:
        mlflow.set_tracking_uri(mlflow_uri)
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        experiment_id = (
            experiment.experiment_id
            if experiment is not None
            else mlflow.create_experiment(self.experiment_name)
        )
        self.experiment_id = experiment_id

        # Search all runs and extract only the "R####" parts to determine the highest run number
        all_runs = mlflow.search_runs(experiment_ids=[experiment_id])

        # Use re to extract the run numbers ("R####")
        run_numbers = (
            all_runs["tags.mlflow.runName"]
            .str.extract(r"R(\d{4})")[0]
            .dropna()
            .astype(int)
        )
        next_run_num = 1 if run_numbers.empty else run_numbers.max() + 1

        # Format run name with just the run number in "R####" format
        self.run_name = f"R{next_run_num:04d}"

        CONSOLE.log(f"MLflow setup complete. Run name: {self.run_name}")
