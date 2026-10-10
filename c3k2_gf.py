"""C3k2-GF reconstructed from the manuscript equations and CSP structure.

This is a reference implementation, not the code used for the reported paper
experiments. GFCore implements the stated fusion without an additional identity
addition. The outer C2f concatenation follows the ordinary C2f/C3k2 definitions
inspected in the supplied project. The inner CSP follows the manuscript's single
1x1 projection followed by a channel split into a processed path and a bypass.

Implementation conventions not specified by the manuscript:
* Three channel partitions are nearly equal; remainders go to the earlier
  partitions. No channels are discarded or padded, and each partition is nonempty.
* The depthwise branches use bare, bias-free convolutions, stride 1 and zero
  padding. No normalization, activation, FFT or extra attention is inserted.
* Local averaging uses stride 1, padding 3 and excludes padded values from the
  average. This border convention needs comparison with the experimental code.
* GF_Bottleneck uses identity projections when channel dimensions already match.
  Otherwise, 1x1 ConvBNAct projections adapt input/hidden/output channels. ``g``
  applies to these optional projections; the GF branches remain depthwise.
  With no adapter, ``g`` introduces no extra operator.
* ``shortcut=False`` is the equation-faithful default. Explicit ``shortcut=True``
  adds a standard optional bottleneck identity branch when c1 == c2. That branch
  is outside the manuscript's X1 * X2 + X1 + X2 fusion expression.
* ``c3k=True`` uses two GF units in a single-projection inner CSP, taking the
  repeat count from the ordinary C3k2 convention. Each inner CSP has hidden
  ratio 0.5. Exact repeat counts, projection widths and
  shortcuts still need checking against the authors' experiment configuration.

The public constructors are intentionally free of Ultralytics imports. ``ConvBNAct``
is the package's shared 1x1 convolution / batch-normalization / SiLU wrapper.
"""

from __future__ import annotations

import math
import operator

import torch
from torch import Tensor, nn

from .common import ConvBNAct

__all__ = ["TNSG", "GFCore", "GF_Bottleneck", "C3k_GF", "C3k2_GF", "C3k2GF"]


def _positive_int(value: int, name: str) -> int:
    """Validate channel/group counts without silently truncating float inputs."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if result <= 0:
        raise ValueError(f"{name} must be positive, got {result}")
    return result


def _repeat_count(n: int) -> int:
    if isinstance(n, bool):
        raise TypeError("n must be a nonnegative integer, not bool")
    try:
        result = operator.index(n)
    except TypeError as exc:
        raise TypeError("n must be a nonnegative integer") from exc
    if result < 0:
        raise ValueError("n must be nonnegative")
    return result


def _hidden_channels(c2: int, e: float) -> int:
    e = float(e)
    if not math.isfinite(e) or e <= 0:
        raise ValueError("e must be a finite positive expansion ratio")
    hidden = int(c2 * e)
    if hidden < 3:
        raise ValueError(
            f"int(c2 * e)={hidden} is too small for three nonempty GF groups; "
            "increase the channels or expansion ratio"
        )
    return hidden


def _adapter(c1: int, c2: int, g: int) -> nn.Module:
    """Adapt channel widths only where the core's channel-preserving rule needs it."""
    if c1 == c2:
        return nn.Identity()
    if c1 % g or c2 % g:
        raise ValueError(
            f"adapter channels ({c1}, {c2}) must both be divisible by g={g}"
        )
    return ConvBNAct(c1, c2, k=1, s=1, g=g)


class TNSG(nn.Module):
    """Return G = sigmoid(1 - AvgPool7x7(X)) for a BCHW feature tensor.

    The returned gate has the input shape. Pooling excludes padding from its
    divisor; ``GFCore`` applies the gate to the input exactly once.
    """

    def __init__(self) -> None:
        super().__init__()
        self.avg_pool = nn.AvgPool2d(
            kernel_size=7, stride=1, padding=3, count_include_pad=False
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError("TNSG expects a BCHW tensor")
        return torch.sigmoid(1.0 - self.avg_pool(x))


class GFCore(nn.Module):
    """Channel-preserving gate/multiscale fusion with the exact manuscript equation.

    X1 = X * sigmoid(1 - AvgPool7(X));
    X2 = Concat(DW3(x_a), DW5(x_b), DW7(x_c));
    Y = X1 * X2 + X1 + X2.

    At least three channels are needed. Nearly equal contiguous partitions support
    channel counts not divisible by three, including ordinary YOLO feature widths.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = _positive_int(channels, "channels")
        if self.channels < 3:
            raise ValueError("GFCore requires at least three channels")
        quotient, remainder = divmod(self.channels, 3)
        self.split_sizes = tuple(
            quotient + int(i < remainder) for i in range(3)
        )
        self.gate = TNSG()
        self.depthwise = nn.ModuleList(
            nn.Conv2d(
                width,
                width,
                kernel_size=kernel,
                stride=1,
                padding=kernel // 2,
                groups=width,
                bias=False,
            )
            for width, kernel in zip(self.split_sizes, (3, 5, 7))
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(
                f"GFCore expects BCHW with {self.channels} channels, "
                f"got shape {tuple(x.shape)}"
            )
        x1 = x * self.gate(x)
        partitions = x.split(self.split_sizes, dim=1)
        x2 = torch.cat(
            [conv(part) for conv, part in zip(self.depthwise, partitions)], dim=1
        )
        return x1 * x2 + x1 + x2


class GF_Bottleneck(nn.Module):
    """Wrap ``GFCore`` with optional channel adapters and an explicit shortcut.

    Defaults preserve the manuscript fusion when c1 == c2. Width adaptation and
    ``shortcut=True`` are reconstruction conventions documented at module level,
    rather than evidence about the original training implementation.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = False,
        g: int = 1,
        e: float = 1.0,
    ) -> None:
        super().__init__()
        c1 = _positive_int(c1, "c1")
        c2 = _positive_int(c2, "c2")
        g = _positive_int(g, "g")
        self.c = _hidden_channels(c2, e)
        self.cv1 = _adapter(c1, self.c, g)
        self.gf = GFCore(self.c)
        self.cv2 = _adapter(self.c, c2, g)
        self.add = bool(shortcut) and c1 == c2

    def forward(self, x: Tensor) -> Tensor:
        y = self.cv2(self.gf(self.cv1(x)))
        return x + y if self.add else y


class C3k_GF(nn.Module):
    """Single-projection inner CSP matching the manuscript's split computation.

    One 1x1 projection yields 2 * hidden channels, then a channel split forms
    two paths. The first passes through n GF units; the second is a bypass.
    Their final outputs are concatenated and projected to c2. The outer C2f,
    unlike this inner CSP, retains all intermediate outputs.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
    ) -> None:
        super().__init__()
        c1 = _positive_int(c1, "c1")
        c2 = _positive_int(c2, "c2")
        n = _repeat_count(n)
        g = _positive_int(g, "g")
        self.c = _hidden_channels(c2, e)
        self.cv1 = ConvBNAct(c1, 2 * self.c, k=1)
        self.cv2 = ConvBNAct(2 * self.c, c2, k=1)
        self.m = nn.Sequential(
            *(GF_Bottleneck(self.c, self.c, shortcut, g, e=1.0) for _ in range(n))
        )

    def forward(self, x: Tensor) -> Tensor:
        processed, bypass = self.cv1(x).split((self.c, self.c), dim=1)
        return self.cv2(torch.cat((self.m(processed), bypass), dim=1))


class C3k2_GF(nn.Module):
    """C2f-style outer C3k2 wrapper with GF or two-unit C3k-GF inner blocks.

    The outer projection makes two hidden features. Each of the n blocks consumes
    the last feature, while all (2 + n) features are retained for final concatenation.
    ``c3k=True`` uses two GF bottlenecks inside each single-projection CSP subunit.
    ``shortcut`` defaults to False to keep the GF fusion equation unchanged.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        g: int = 1,
        shortcut: bool = False,
    ) -> None:
        super().__init__()
        c1 = _positive_int(c1, "c1")
        c2 = _positive_int(c2, "c2")
        n = _repeat_count(n)
        g = _positive_int(g, "g")
        self.c = _hidden_channels(c2, e)
        self.cv1 = ConvBNAct(c1, 2 * self.c, k=1)
        self.cv2 = ConvBNAct((2 + n) * self.c, c2, k=1)
        self.m = nn.ModuleList(
            C3k_GF(self.c, self.c, n=2, shortcut=shortcut, g=g, e=0.5)
            if c3k
            else GF_Bottleneck(self.c, self.c, shortcut=shortcut, g=g, e=1.0)
            for _ in range(n)
        )

    def forward(self, x: Tensor) -> Tensor:
        y = list(self.cv1(x).chunk(2, dim=1))
        for block in self.m:
            y.append(block(y[-1]))
        return self.cv2(torch.cat(y, dim=1))

    def forward_split(self, x: Tensor) -> Tensor:
        """Equivalent explicit split path, matching the ordinary C2f interface."""
        y = list(self.cv1(x).split((self.c, self.c), dim=1))
        for block in self.m:
            y.append(block(y[-1]))
        return self.cv2(torch.cat(y, dim=1))


C3k2GF = C3k2_GF
