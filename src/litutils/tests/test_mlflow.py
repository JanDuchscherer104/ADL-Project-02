import sys
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from litflow.configs.mlflow import MLflowConfig


class TestMLflowConfig(unittest.TestCase):
    @patch("mlflow.set_tracking_uri")
    @patch("mlflow.get_experiment_by_name")
    @patch("mlflow.create_experiment")
    @patch("mlflow.search_runs")
    def test_setup_mlflow_first_run(
        self,
        mock_search_runs,
        mock_create_experiment,
        mock_get_experiment,
        mock_set_tracking_uri,
    ):
        # Mock MLflow functions
        mock_set_tracking_uri.return_value = None
        mock_get_experiment.return_value = None  # Simulate that no experiment exists
        mock_create_experiment.return_value = "1"  # Simulate new experiment creation
        # Return an empty DataFrame with the expected column structure
        mock_search_runs.return_value = pd.DataFrame(columns=["tags.mlflow.runName"])

        config = MLflowConfig()
        config.setup_mlflow("fake_mlflow_uri")

        # Assertions
        mock_set_tracking_uri.assert_called_once_with("fake_mlflow_uri")
        mock_create_experiment.assert_called_once_with(config.experiment_name)
        self.assertEqual(config.run_name, "R0001")
        self.assertEqual(config.experiment_id, "1")

    @patch("mlflow.set_tracking_uri")
    @patch("mlflow.get_experiment_by_name")
    @patch("mlflow.search_runs")
    def test_setup_mlflow_existing_runs(
        self, mock_search_runs, mock_get_experiment, mock_set_tracking_uri
    ):
        # Mock MLflow functions
        mock_set_tracking_uri.return_value = None
        mock_get_experiment.return_value = MagicMock(experiment_id="1")

        # Simulate previous runs with run names like "R0001", "R0002", "R0003"
        previous_runs = pd.DataFrame(
            {"tags.mlflow.runName": ["R0001", "R0002", "R0003"]}
        )
        mock_search_runs.return_value = previous_runs

        config = MLflowConfig()
        config.setup_mlflow("fake_mlflow_uri")

        # Assertions
        mock_set_tracking_uri.assert_called_once_with("fake_mlflow_uri")
        self.assertEqual(
            config.run_name, "R0004"
        )  # Run name should increment to the next available
        self.assertEqual(config.experiment_id, "1")


if __name__ == "__main__":
    unittest.main()
