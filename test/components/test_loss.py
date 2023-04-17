import torch
from tmlc.components.loss import inverse_frequency_weighted

class InverseRequencyWeightedTests:
    def test_class_count(self):
        labels = torch.tensor([[1, 1, 0], [0, 1, 1], [1, 1, 0], [0, 0, 1]])
        expected_class_count = torch.tensor([2, 3, 2])
        result = inverse_frequency_weighted(labels)
        class_count = torch.sum(labels, axis=0)
        assert torch.equal(class_count, expected_class_count)


    def test_class_frequency(self):
        labels = torch.tensor([[1, 1, 0], [0, 1, 1], [1, 1, 0], [0, 0, 1]])
        expected_class_freq = torch.tensor([0.6, 0.8, 0.6])
        result = inverse_frequency_weighted(labels)
        class_count = torch.sum(labels, axis=0)
        num_samples, _ = labels.shape
        class_freq = (class_count + 1) / (num_samples + 1)
        assert torch.allclose(class_freq, expected_class_freq, atol=1e-6)


    def test_class_weights(self):
        labels = torch.tensor([[1, 1, 0], [0, 1, 1], [1, 1, 0], [0, 0, 1]])
        expected_class_weights = torch.tensor([1.6667, 1.25, 1.6667])
        result = inverse_frequency_weighted(labels)
        assert torch.allclose(result["pos_weight"], expected_class_weights, atol=1e-4)


    def test_empty_labels(self):
        labels = torch.tensor([[]])
        result = inverse_frequency_weighted(labels)
        assert torch.equal(result["pos_weight"], torch.tensor([]))


    def test_no_samples(self):
        labels = torch.tensor([[]])
        expected_class_weights = torch.tensor([])
        result = inverse_frequency_weighted(labels)
        assert torch.equal(result["pos_weight"], expected_class_weights)
