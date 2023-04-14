import pytest
import torch

from tmlc.components.loss import loss_fn_weighted


class TestLossFunction:
    @pytest.fixture
    def binary_labels(self):
        # create binary labels tensor of shape (num_samples, num_classes)
        return torch.tensor([[1, 0], [0, 1], [1, 0]])

    def test_loss_fn_weighted(self, binary_labels):
        loss_fn = loss_fn_weighted(binary_labels)
        assert isinstance(loss_fn, torch.nn.BCEWithLogitsLoss)

    def test_loss_fn_weighted_close(self):
        # create a test tensor of labels
        labels = torch.tensor([[1, 1, 0], [0, 1, 1], [1, 1, 0], [0, 0, 1]])

        expected_weights = torch.tensor([2.0, 1.3333, 2.0], dtype=torch.float32)

        # calculate the actual class weights using the function
        loss_fn = loss_fn_weighted(labels)
        actual_weights = loss_fn.pos_weight

        # check that the expected and actual weights are equal
        assert torch.allclose(actual_weights, expected_weights, rtol=1e-4, atol=1e-4)

    def test_loss_fn_weighted_output(self, binary_labels):
        loss_fn = loss_fn_weighted(binary_labels)
        preds = torch.tensor([[0.5, -0.5], [-0.5, 0.5], [0.5, -0.5]])
        targets = preds
        output = loss_fn(preds, targets)
        assert isinstance(output, torch.Tensor)
