from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class GeneralizedClassifier(ABC, nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, pooled_output: torch.Tensor, classifier_additional: torch.Tensor) -> torch.Tensor:
        pass
