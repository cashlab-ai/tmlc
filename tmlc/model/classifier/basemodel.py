import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class GeneralizedClassifier(ABC, nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
