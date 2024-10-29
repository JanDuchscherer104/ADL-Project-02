import unittest

import torch
from torch import Tensor

from ingredient_classifier import ClassifierType, IngredientClassifierParams


# Unit Test / Sanity Check for the LitImageClassifierModule
class TestLitImageClassifierModule(unittest.TestCase):
    def setUp(self):
        self.params = IngredientClassifierParams(
            model=ClassifierType.ALEXNET, num_classes=10
        )
        self.model = self.params.setup_target()

    def test_forward_pass(self):
        """Test forward pass with dummy data."""
        batch_size = 4
        dummy_input = torch.randn(batch_size, 3, 224, 224)
        output = self.model(dummy_input)
        self.assertEqual(output.shape, (batch_size, self.params.num_classes))

    def test_training_step(self):
        """Test training step to ensure loss calculation works."""
        batch_size = 4
        dummy_input = torch.randn(batch_size, 3, 224, 224)
        dummy_labels = torch.randint(0, self.params.num_classes, (batch_size,))
        batch = (dummy_input, dummy_labels)
        loss = self.model.training_step(batch, 0)
        self.assertIsInstance(loss, Tensor)
        self.assertGreaterEqual(loss.item(), 0.0)

    def test_validation_step(self):
        """Test validation step to ensure metrics calculation works."""
        batch_size = 4
        dummy_input = torch.randn(batch_size, 3, 224, 224)
        dummy_labels = torch.randint(0, self.params.num_classes, (batch_size,))
        batch = (dummy_input, dummy_labels)
        self.model.validation_step(batch, 0)  # Ensure no exceptions are thrown


if __name__ == "__main__":
    unittest.main(argv=[""], exit=False)
