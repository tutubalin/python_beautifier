"""Array indexing in the wild: the slicing that fills NumPy and PyTorch code bases.

Every subscript below mixes ``:``, ``...``, ``None``, negative numbers and steps.
The beautifier marks each comma separated term inside the brackets and spells out
what it does, so that ``tokens[:, n:, ...]`` can be read instead of decoded.

This file is for reading only; nothing in it is executed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch


def split_text_and_image(tokens: torch.Tensor, num_txt_tokens: int):
    """Split a joint token sequence into its text and image parts.

    Args:
        tokens: Sequence of shape ``(batch, length, dim)``; the text tokens come first.
        num_txt_tokens: How many tokens at the start of the sequence are text.

    Returns:
        The text tokens and the image tokens.
    """
    txt = tokens[:, :num_txt_tokens]
    img = tokens[:, num_txt_tokens:, ...]
    return txt, img


def last_token_logits(logits: torch.Tensor) -> torch.Tensor:
    """Logits of the final position of every sequence in the batch."""
    return logits[:, -1, :]


def add_axes(image: np.ndarray) -> np.ndarray:
    """Turn an ``(H, W)`` image into ``(1, H, W, 1)``: a batch axis and a channel axis."""
    return image[None, ..., None]


def causal_mask(length: int) -> torch.Tensor:
    """Lower-triangular attention mask shaped to broadcast over batch and heads."""
    mask = torch.tril(torch.ones(length, length, dtype=torch.bool))
    return mask[None, None, :, :]


def pairwise_differences(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """All differences ``a_i - b_j`` for ``a`` of shape (N, D) and ``b`` of shape (M, D)."""
    return a[:, None, :] - b[None, :, :]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate pairs of features by 90 degrees, as rotary position embeddings need."""
    even = x[..., ::2]
    odd = x[..., 1::2]
    return torch.stack((-odd, even), dim=-1).flatten(-2)


def flip_horizontal(image: np.ndarray) -> np.ndarray:
    """Mirror an image left to right, whatever trailing channel axes it has."""
    return image[:, ::-1, ...]


def center_crop(image: np.ndarray, size: int) -> np.ndarray:
    """Cut a ``size`` by ``size`` square out of the middle of ``image``."""
    top = (image.shape[0] - size) // 2
    left = (image.shape[1] - size) // 2
    return image[top : top + size, left : left + size]


def normalise_channels(batch: np.ndarray) -> np.ndarray:
    """Give every colour channel of a batch of images zero mean and unit variance."""
    mean = batch.mean(axis=(0, 1, 2))
    std = batch.std(axis=(0, 1, 2)) + 1e-8
    return (batch - mean[None, None, None, :]) / std[None, None, None, :]


def real_positions(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Keep only the real (not padded) positions of a padded batch."""
    positions = torch.arange(x.shape[1])[None, :]
    keep = positions < lengths[:, None]
    return x[keep]


def zero_outliers(values: np.ndarray, limit: float) -> np.ndarray:
    """Zero every row whose first column is above ``limit``."""
    values = values.copy()
    values[values[:, 0] > limit, :] = 0
    return values


def has_signal(frames: np.ndarray, threshold: float) -> bool:
    """True when the first channel of any frame rises above ``threshold``."""
    if frames[..., 0].max() > threshold:
        return True
    for frame in frames[::2, ...]:
        if frame[:, :, 0].sum() > 0:
            return True
    return False


def clamp_borders(volume: np.ndarray) -> np.ndarray:
    """Zero the outermost layer of a 3-D volume, in place."""
    volume[0, :, :] = 0
    volume[-1, :, :] = 0
    volume[:, [0, -1], :] = 0
    volume[..., 0] = 0
    return volume


def top_rows(table: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """The first ``n`` rows of a table, without its first column (position based)."""
    return table.iloc[:n, 1:]


def price_columns(table: pd.DataFrame) -> pd.DataFrame:
    """Columns ``price`` through ``tax`` of every row (label based: the end is included)."""
    return table.loc[:, "price":"tax"]
