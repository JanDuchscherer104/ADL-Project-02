import unittest

import optuna

from litutils import Optimizable


class TestOptunaOptimizable(unittest.TestCase):
    def setUp(self):
        self.trial = optuna.trial.FixedTrial(
            {"int_param": 5, "float_param": 0.5, "bool_param": True, "cat_param": "a"}
        )

    def test_int_target(self):
        config = Optimizable(start=1, end=10, step=1, target=int)
        result = config.setup_target("int_param", self.trial)
        self.assertIsInstance(result, int)

    def test_float_target(self):
        config = Optimizable(start=0.1, end=1.0, target=float)
        result = config.setup_target("float_param", self.trial)
        self.assertIsInstance(result, float)

    def test_bool_target(self):
        config = Optimizable(default=True, target=bool)
        result = config.setup_target("bool_param", self.trial)
        self.assertIsInstance(result, bool)

    def test_cat_target(self):
        config = Optimizable(categories=["a", "b", "c"], target=str)
        result = config.setup_target("cat_param", self.trial)
        self.assertIsInstance(result, str)

    def test_int_target_missing_values(self):
        config = Optimizable(target=int)
        with self.assertRaises(ValueError):
            config.setup_target("int_param", self.trial)

    def test_float_target_missing_values(self):
        config = Optimizable(target=float)
        with self.assertRaises(ValueError):
            config.setup_target("float_param", self.trial)

    def test_cat_target_missing_values(self):
        config = Optimizable(target=str)
        with self.assertRaises(ValueError):
            config.setup_target("cat_param", self.trial)


if __name__ == "__main__":
    unittest.main()
