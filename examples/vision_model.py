"""A small image classifier that demonstrates the static model-schema view.

Input size is stated in the ``forward`` docstring so the diagram can carry exact sizes through
Conv2d, pooling, flatten and Linear. The file is a documentation fixture: torch is never imported
or run by the beautifier.
"""
import torch
from torch import nn


class TorchModel(nn.Module):
    """An intermediate base class, so descendants are detected transitively."""

    def __init__(self):
        super().__init__()


class TinyVisionNet(TorchModel):
    """A compact CNN; the schema expands the Sequential block and follows the residual join.

    Args:
        x: A batch of RGB images shaped ``(B, 3, 32, 32)``.
    """

    def __init__(self, classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.shortcut = nn.Conv2d(3, 32, kernel_size=1, stride=2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(32, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process an image batch. ``x`` has shape ``(B, 3, 32, 32)``."""
        main = self.features(x)
        skip = self.shortcut(x)
        merged = main + skip
        pooled = self.global_pool(merged)
        flattened = torch.flatten(pooled, 1)
        return self.classifier(flattened)


class KVProjection(nn.Module):
    """A module with two content-detected entrypoints rather than ``forward``."""

    def __init__(self):
        super().__init__()
        self.kv_extract = nn.Linear(64, 128)
        self.kv_cached = nn.Linear(64, 64)

    def forward_kv_extract(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Extract concatenated key/value features from ``(B, T, 64)`` states."""
        return self.kv_extract(hidden_states)

    def forward_kv_cached(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project a cached key/value state of shape ``(B, T, 64)``."""
        return self.kv_cached(hidden_states)


class LooksLikeATorchModel:
    """Detected structurally: a forward method calls named layer-like attributes."""

    def __init__(self):
        self.projection = Linear(128, 32)
        self.activation = ReLU()

    def forward(self, x):
        return self.activation(self.projection(x))
