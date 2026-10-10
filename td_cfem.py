"""Paper-grounded TD-CFEM, with explicit choices for unspecified settings.

Source: ``outputs/Applied-Intelligence/manuscript.tex``, Sect. ``sec:tdcfem``
and Figs. 6--7. This is a reconstruction, not the authors' original training
implementation. The inspected Ultralytics ``nn`` tree did not contain a
TD-CFEM/PFSG implementation.

Paper-defined operations are the fixed Laplacian, CBR spatial gate, ECA,
four equal channel groups with 16 phase-adjustable cosine candidates,
Gumbel-Softmax training selection / deterministic hard inference selection,
normalized squared descriptor energies, logarithmic group gates, a two-layer
frequency MLP, joint multiplication, and a ratio-2 residual transform.

Implementation conventions, not recovered experimental settings:
* candidate frequencies are the visible Cartesian 4 x 4 grid below;
* cosine projection uses the raw spatial sum by default, without DCT scaling;
* the 16 selection logits and 16 pairs of phases are shared across groups,
  matching the shared W_base/alpha notation; per-group parameters are optional;
* phases/logits start at zero; temperature=1, straight-through hard training;
* energy slope=1, energy bias=0, epsilon=1e-6, MLP reduction=16;
* ECA gamma=2 and b=1, rounding down to an odd kernel as the stated floor_odd
  rule; an explicit kernel or upward rounding can be requested;
* Laplacian padding is zero; the spatial CBR preserves C channels;
* feature convolutions have biases and no BN by default, so Swish occurs only
  after the final projection as in Eq. p204. Optional BN is an explicit variant.

No FFT, fixed DCT replacement, pooling to a reference resolution, or additional
frequency normalization is silently introduced. H and W are taken from the
current tensor; odd and nonsquare resolutions are supported. As for standard
PyTorch convolution modules, move the module to the input device/dtype with
``module.to(device=..., dtype=...)``. Frequency projection, energy, and its MLP
accumulate in float32 for float16/bfloat16 inputs to avoid half-precision
overflow; float64 is preserved. Output gates return to the input dtype.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F


DEFAULT_FREQUENCIES: tuple[tuple[int, int], ...] = tuple(
    (u, v) for u in range(4) for v in range(4)
)


def _channels(channels: int) -> int:
    if isinstance(channels, bool) or not isinstance(channels, int) or channels <= 0:
        raise ValueError("channels must be a positive integer")
    return channels


def _accumulation_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.float64 if dtype == torch.float64 else torch.float32


def _input_check(x: Tensor, channels: int) -> None:
    if x.ndim != 4 or x.shape[1] != channels:
        raise ValueError(f"expected BCHW with C={channels}, got {tuple(x.shape)}")
    if min(x.shape[-2:]) < 1:
        raise ValueError("spatial dimensions must be positive")
    if not x.is_floating_point():
        raise TypeError("features must use a floating-point dtype")


class ECA(nn.Module):
    """GAP -> local Conv1D -> sigmoid, producing B x C x 1 x 1."""

    def __init__(
        self,
        channels: int,
        kernel_size: int | None = None,
        gamma: float = 2.0,
        b: float = 1.0,
        odd_rounding: Literal["down", "up"] = "down",
    ) -> None:
        super().__init__()
        self.channels = _channels(channels)
        if not math.isfinite(gamma) or gamma <= 0 or not math.isfinite(b):
            raise ValueError("ECA gamma must be finite and positive; b must be finite")
        if odd_rounding not in ("down", "up"):
            raise ValueError("odd_rounding must be 'down' or 'up'")
        if kernel_size is None:
            kernel_size = max(1, math.floor((math.log2(channels) + b) / gamma))
            if kernel_size % 2 == 0:
                kernel_size += -1 if odd_rounding == "down" else 1
        if not isinstance(kernel_size, int) or kernel_size < 1 or kernel_size % 2 != 1:
            raise ValueError("ECA kernel_size must be a positive odd integer")
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        _input_check(x, self.channels)
        descriptor = x.mean(dim=(-2, -1)).unsqueeze(1)
        return self.conv(descriptor).sigmoid().squeeze(1).unsqueeze(-1).unsqueeze(-1)


class LaplacianSpatialGate(nn.Module):
    """Fixed channelwise Laplacian -> learned 3x3 CBR -> sigmoid."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = _channels(channels)
        kernel = torch.tensor([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]])
        self.register_buffer("laplacian", kernel.view(1, 1, 3, 3).repeat(channels, 1, 1, 1))
        self.cbr = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        _input_check(x, self.channels)
        edges = F.conv2d(x, self.laplacian.to(dtype=x.dtype), padding=1, groups=self.channels)
        return self.cbr(edges).sigmoid()


class PFD(nn.Module):
    """Four-group phase-adjustable cosine projection and candidate selection.

    ``forward`` returns selected descriptors with shape B x 4 x (C/4).
    ``return_aux=True`` additionally returns all descriptors B x 4 x (C/4) x 16
    and selection coefficients 4 x 16, enabling direct formula checks.
    Shared selection samples one candidate distribution for the entire batch
    and all groups, since alpha_k is a learned parameter rather than an
    input-dependent predictor. In eval mode ties select the first candidate.
    """

    def __init__(
        self,
        channels: int,
        frequencies: Sequence[tuple[int, int]] = DEFAULT_FREQUENCIES,
        temperature: float = 1.0,
        train_hard: bool = True,
        projection_normalization: Literal["none", "spatial_mean", "l2"] = "none",
        per_group_selection: bool = False,
        per_group_phase: bool = False,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.channels = _channels(channels)
        if channels % 4:
            raise ValueError("PFD requires channels divisible by four")
        pairs = tuple(tuple(pair) for pair in frequencies)
        if len(pairs) != 16 or any(
            len(pair) != 2 or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in pair)
            for pair in pairs
        ):
            raise ValueError("frequencies must contain exactly 16 nonnegative integer (u, v) pairs")
        if len(set(pairs)) != 16:
            raise ValueError("frequency candidates must be distinct")
        if projection_normalization not in ("none", "spatial_mean", "l2"):
            raise ValueError("projection_normalization must be 'none', 'spatial_mean', or 'l2'")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive")
        self.groups = 4
        self.candidates = 16
        self.group_channels = channels // 4
        self.temperature = float(temperature)
        self.train_hard = bool(train_hard)
        self.projection_normalization = projection_normalization
        self.per_group_selection = bool(per_group_selection)
        self.per_group_phase = bool(per_group_phase)
        self.eps = float(eps)
        self.register_buffer("frequency_indices", torch.tensor(pairs, dtype=torch.int64))
        self.phase = nn.Parameter(torch.zeros((4, 16, 2) if per_group_phase else (16, 2)))
        self.logits = nn.Parameter(torch.zeros((4, 16) if per_group_selection else (16,)))

    def basis(self, height: int, width: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Return 4 x 16 x (H W) bases, flattened using h*W+w."""
        if height < 1 or width < 1:
            raise ValueError("height and width must be positive")
        dtype = _accumulation_dtype(dtype)
        h = torch.arange(height, device=device, dtype=dtype)
        w = torch.arange(width, device=device, dtype=dtype)
        uv = self.frequency_indices.to(device=device, dtype=dtype)
        phase = self.phase.to(device=device, dtype=dtype)
        if not self.per_group_phase:
            phase = phase.unsqueeze(0).expand(4, -1, -1)
        horizontal = torch.cos(
            math.pi * uv[None, :, 0, None] * (2 * h[None, None, :] + 1) / (2 * height)
            + phase[:, :, 0, None]
        )
        vertical = torch.cos(
            math.pi * uv[None, :, 1, None] * (2 * w[None, None, :] + 1) / (2 * width)
            + phase[:, :, 1, None]
        )
        basis = (horizontal.unsqueeze(-1) * vertical.unsqueeze(-2)).flatten(-2)
        if self.projection_normalization == "spatial_mean":
            basis = basis / (height * width)
        elif self.projection_normalization == "l2":
            basis = basis / basis.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        return basis

    def selection(self, *, dtype: torch.dtype) -> Tensor:
        """Differentiable Gumbel selection in train; deterministic argmax in eval."""
        logits = self.logits.to(dtype=dtype)
        if self.training:
            weights = F.gumbel_softmax(logits, tau=self.temperature, hard=self.train_hard, dim=-1)
        else:
            weights = F.one_hot(logits.argmax(dim=-1), num_classes=16).to(dtype=dtype)
        if not self.per_group_selection:
            weights = weights.unsqueeze(0).expand(4, -1)
        return weights

    def forward(self, x: Tensor, return_aux: bool = False):
        _input_check(x, self.channels)
        b, _, h, w = x.shape
        dtype = _accumulation_dtype(x.dtype)
        # Disable external autocast for cosine, projection, and discrete weights.
        with torch.autocast(device_type=x.device.type, enabled=False):
            basis = self.basis(h, w, device=x.device, dtype=dtype)
            flattened = x.to(dtype=dtype).reshape(b, 4, self.group_channels, h * w)
            all_descriptors = torch.einsum("bgcn,gkn->bgck", flattened, basis)
            weights = self.selection(dtype=dtype)
            selected = (all_descriptors * weights[None, :, None, :]).sum(dim=-1)
        if return_aux:
            return selected, all_descriptors, weights
        return selected


class EAFS(nn.Module):
    """Relative squared group energy and learnable logarithmic sigmoid gates."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive")
        self.eps = float(eps)
        self.slope = nn.Parameter(torch.ones(4))
        self.bias = nn.Parameter(torch.zeros(4))

    def forward(self, descriptors: Tensor, return_aux: bool = False):
        if descriptors.ndim != 3 or descriptors.shape[1] != 4 or descriptors.shape[2] < 1:
            raise ValueError("EAFS expects B x 4 x group_channels descriptors")
        dtype = _accumulation_dtype(descriptors.dtype)
        with torch.autocast(device_type=descriptors.device.type, enabled=False):
            d = descriptors.to(dtype=dtype)
            energy = d.square().sum(dim=-1)
            relative = energy / (energy.sum(dim=-1, keepdim=True) + self.eps)
            gates = (
                self.slope.to(dtype=dtype) * (relative + self.eps).log()
                + self.bias.to(dtype=dtype)
            ).sigmoid()
            weighted = d * gates.unsqueeze(-1)
        if return_aux:
            return weighted, energy, relative, gates
        return weighted


class PFSG(nn.Module):
    """PFD -> energy weighting -> channel concat -> two-layer MLP -> sigmoid."""

    def __init__(
        self,
        channels: int,
        frequencies: Sequence[tuple[int, int]] = DEFAULT_FREQUENCIES,
        temperature: float = 1.0,
        train_hard: bool = True,
        projection_normalization: Literal["none", "spatial_mean", "l2"] = "none",
        per_group_selection: bool = False,
        per_group_phase: bool = False,
        eps: float = 1e-6,
        reduction: int = 16,
        mlp_bias: bool = True,
    ) -> None:
        super().__init__()
        self.channels = _channels(channels)
        if not isinstance(reduction, int) or reduction < 1:
            raise ValueError("reduction must be a positive integer")
        self.pfd = PFD(
            channels, frequencies, temperature, train_hard, projection_normalization,
            per_group_selection, per_group_phase, eps,
        )
        self.eafs = EAFS(eps)
        self.reduction = reduction
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden, bias=mlp_bias)
        self.fc2 = nn.Linear(hidden, channels, bias=mlp_bias)

    @staticmethod
    def _linear_accumulated(x: Tensor, layer: nn.Linear) -> Tensor:
        bias = None if layer.bias is None else layer.bias.to(dtype=x.dtype)
        return F.linear(x, layer.weight.to(dtype=x.dtype), bias)

    def forward(self, x: Tensor, return_aux: bool = False):
        selected, all_descriptors, selection = self.pfd(x, return_aux=True)
        weighted, energy, relative, group_gates = self.eafs(selected, return_aux=True)
        with torch.autocast(device_type=x.device.type, enabled=False):
            descriptor = weighted.reshape(x.shape[0], self.channels)
            hidden = F.relu(self._linear_accumulated(descriptor, self.fc1))
            gate = self._linear_accumulated(hidden, self.fc2).sigmoid()
        gate = gate.to(dtype=x.dtype).unsqueeze(-1).unsqueeze(-1)
        if return_aux:
            return gate, {
                "selected_descriptors": selected,
                "all_descriptors": all_descriptors,
                "selection": selection,
                "energy": energy,
                "relative_energy": relative,
                "group_gates": group_gates,
            }
        return gate


class TD_CFEM(nn.Module):
    """Tri-domain gating followed by a ratio-2 inverted residual transform.

    Input and output are B x C x H x W. The paper's identity shortcut requires
    c2 == c1; a channel-changing shortcut is not silently added. All settings
    left unspecified by the manuscript are constructor arguments or clearly
    stated conventions in the module docstring.
    """

    def __init__(
        self,
        c1: int,
        c2: int | None = None,
        *,
        frequencies: Sequence[tuple[int, int]] = DEFAULT_FREQUENCIES,
        temperature: float = 1.0,
        train_hard: bool = True,
        projection_normalization: Literal["none", "spatial_mean", "l2"] = "none",
        per_group_selection: bool = False,
        per_group_phase: bool = False,
        eps: float = 1e-6,
        reduction: int = 16,
        mlp_bias: bool = True,
        eca_kernel: int | None = None,
        eca_gamma: float = 2.0,
        eca_b: float = 1.0,
        eca_odd_rounding: Literal["down", "up"] = "down",
        feature_batchnorm: bool = False,
        feature_bias: bool = True,
    ) -> None:
        super().__init__()
        self.channels = _channels(c1)
        if c2 is not None and c2 != c1:
            raise ValueError("TD_CFEM requires c2 == c1 for its original-input identity shortcut")
        if c1 % 4:
            raise ValueError("TD_CFEM requires channels divisible by four")
        self.spatial = LaplacianSpatialGate(c1)
        self.channel = ECA(c1, eca_kernel, eca_gamma, eca_b, eca_odd_rounding)
        self.frequency = PFSG(
            c1, frequencies, temperature, train_hard, projection_normalization,
            per_group_selection, per_group_phase, eps, reduction, mlp_bias,
        )
        expanded = 2 * c1

        def conv(cin: int, cout: int, k: int, groups: int = 1) -> nn.Module:
            layer = nn.Conv2d(
                cin, cout, k, padding=k // 2, groups=groups,
                bias=feature_bias and not feature_batchnorm,
            )
            return nn.Sequential(layer, nn.BatchNorm2d(cout)) if feature_batchnorm else layer

        self.expand = conv(c1, expanded, 1)
        self.depthwise = conv(expanded, expanded, 3, groups=expanded)
        self.project = conv(expanded, c1, 1)
        self.activation = nn.SiLU(inplace=False)  # Swish only after final projection.

    def forward(self, x: Tensor) -> Tensor:
        _input_check(x, self.channels)
        gated = x * self.spatial(x) * self.channel(x) * self.frequency(x)
        transformed = self.project(self.depthwise(self.expand(gated)))
        return x + self.activation(transformed)


TDCFEM = TD_CFEM

__all__ = [
    "DEFAULT_FREQUENCIES", "ECA", "LaplacianSpatialGate", "PFD", "EAFS",
    "PFSG", "TD_CFEM", "TDCFEM",
]
