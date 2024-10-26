import tempfile
import unittest
from pathlib import Path

from litutils.global_configs.paths import PathConfig


class TestPathConfig(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.default_config = PathConfig(root=Path(self.temp_dir.name))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_default_paths(self):
        self.assertTrue(self.default_config.data.exists())
        self.assertTrue(self.default_config.checkpoints.exists())
        self.assertTrue(self.default_config.tb_logs.exists())
        self.assertTrue(self.default_config.configs.exists())

    def test_custom_paths(self):
        with tempfile.TemporaryDirectory() as custom_root:
            custom_root_path = Path(custom_root)
            config = PathConfig(root=custom_root_path)
            self.assertEqual(config.root, custom_root_path)
            self.assertTrue(config.data.exists())
            self.assertTrue(config.checkpoints.exists())

    def test_absolute_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            absolute_path = Path(temp_dir) / "absolute_path"
            config = PathConfig(data=absolute_path)
            self.assertTrue(config.data.exists())


if __name__ == "__main__":
    unittest.main()
