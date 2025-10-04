##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
# Extended by ChatGPT, 2025-07-11                #
##################################################
"""
This file extends the original NAS-Bench-201 OPS implementation with a
richer operator set aimed at enabling energy‑aware NAS experiments.

Changes & additions
-------------------
1.  Added wider‑kernel normal convolutions (`nor_conv_5x5`).
2.  Added depth‑wise separable convolutions of multiple kernel sizes
   (`dw_conv_{3,5,7}x{3,5,7}`).
3.  Added grouped convolution (`group_conv_3x3`).
4.  Added 5×5 average / max pooling.
5.  Removed duplicate `skip_connect` key and unified its behaviour.
6.  Generalised `POOLING` to accept an arbitrary kernel size.
7.  Added simple `Shift` operator (zero‑FLOPs spatial shift).
8.  Declared a public `OPS` alias so `from <file> import OPS` works.
"""

from typing import Callable, Dict
import torch
import torch.nn as nn

# __all__ = [
#     "OPS",  # public mapping: op_name -> nn.Module builder
#     "ResNetBasicblockEnergy",
#     "SearchSpaceNames",
# ]

# -----------------------------------------------------------------------------
#   Helper Layers
# -----------------------------------------------------------------------------

class ReLUConvBN(nn.Module):
    """ReLU → Conv → BN block (with customizable dilation)."""

    def __init__(
        self,
        C_in: int,
        C_out: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int = 1,
        affine: bool = True,
        track_running_stats: bool = True,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                C_in,
                C_out,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(C_out, affine=affine, track_running_stats=track_running_stats),
        )

    def forward(self, x):  # noqa: D401
        return self.op(x)


class SepConv(nn.Module):
    """Depth‑wise separable convolution (single stack)."""

    def __init__(
        self,
        C_in: int,
        C_out: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int = 1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                C_in,
                C_in,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=C_in,
                bias=False,
            ),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine, track_running_stats=track_running_stats),
        )

    def forward(self, x):  # noqa: D401
        return self.op(x)


class DualSepConv(nn.Module):
    """Two‑stack depth‑wise separable convolution (MobileNet‑V2 style)."""

    def __init__(
        self,
        C_in: int,
        C_out: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int = 1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        super().__init__()
        self.op_a = SepConv(
            C_in,
            C_in,
            kernel_size,
            stride,
            padding,
            dilation,
            affine,
            track_running_stats,
        )
        self.op_b = SepConv(
            C_in,
            C_out,
            kernel_size,
            1,
            padding,
            dilation,
            affine,
            track_running_stats,
        )

    def forward(self, x):  # noqa: D401
        x = self.op_a(x)
        x = self.op_b(x)
        return x


class GroupConvBN(nn.Module):
    """Grouped convolution with ReLU + BN wrapper."""

    def __init__(
        self,
        C_in: int,
        C_out: int,
        kernel_size: int,
        stride: int,
        padding: int,
        groups: int = 4,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                C_in,
                C_out,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(C_out, affine=affine, track_running_stats=track_running_stats),
        )

    def forward(self, x):  # noqa: D401
        return self.op(x)


class Shift(nn.Module):
    """Zero-FLOPs spatial shift with zero padding (ONNX/TFLite friendly).
       We split channels into 4 groups and apply {identity, up, down, right}.
    """
    def __init__(self, channels: int) -> None:
        super().__init__()
        assert channels % 4 == 0, "Shift op requires channels divisible by 4"
        self.channels = channels

    @staticmethod
    def _shift_up(t):    # t: [B, G, H, W]
        pad = torch.zeros_like(t[:, :, :1, :])
        return torch.cat([t[:, :, 1:, :], pad], dim=2)

    @staticmethod
    def _shift_down(t):
        pad = torch.zeros_like(t[:, :, :1, :])
        return torch.cat([pad, t[:, :, :-1, :]], dim=2)

    @staticmethod
    def _shift_left(t):
        pad = torch.zeros_like(t[:, :, :, :1])
        return torch.cat([t[:, :, :, 1:], pad], dim=3)

    @staticmethod
    def _shift_right(t):
        pad = torch.zeros_like(t[:, :, :, :1])
        return torch.cat([pad, t[:, :, :, :-1]], dim=3)

    def forward(self, x):
        B, C, H, W = x.shape
        G = C // 4
        xg = x.view(B, G, 4, H, W)
        x0 = xg[:, :, 0, :, :]    # identity
        x1 = xg[:, :, 1, :, :]    # up
        x2 = xg[:, :, 2, :, :]    # down
        x3 = xg[:, :, 3, :, :]    # right

        y0 = x0
        y1 = self._shift_up(x1)
        y2 = self._shift_down(x2)
        y3 = self._shift_right(x3)

        y = torch.cat([y0, y1, y2, y3], dim=1)   # [B, 4G, H, W]
        return y.view(B, C, H, W)


class POOLING(nn.Module):
    """Avg/Max pooling with optional 1×1 pre‑processing conv to match channels."""

    def __init__(
        self,
        C_in: int,
        C_out: int,
        stride: int,
        mode: str,
        affine: bool = True,
        track_running_stats: bool = True,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if C_in == C_out:
            self.preprocess = None
        else:
            self.preprocess = ReLUConvBN(
                C_in, C_out, 1, 1, 0, 1, affine, track_running_stats
            )
        pad = kernel_size // 2
        if mode == "avg":
            self.op = nn.AvgPool2d(
                kernel_size, stride=stride, padding=pad, count_include_pad=False
            )
        elif mode == "max":
            self.op = nn.MaxPool2d(kernel_size, stride=stride, padding=pad)
        else:
            raise ValueError(f"Invalid mode={mode} in POOLING")

    def forward(self, inputs):  # noqa: D401
        x = self.preprocess(inputs) if self.preprocess else inputs
        return self.op(x)


class Identity(nn.Module):
    def forward(self, x):  # noqa: D401
        return x


class Zero(nn.Module):
    """Zero out feature map (optionally with stride)."""

    def __init__(self, C_in: int, C_out: int, stride: int) -> None:
        super().__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.stride = stride
        self.is_zero = True

    def forward(self, x):
        if self.C_in == self.C_out:
            if self.stride == 1:
                return x * 0.0                       # ← 非 in-place
            return x[:, :, :: self.stride, :: self.stride] * 0.0
        shape = list(x.shape); shape[1] = self.C_out
        return x.new_zeros(shape, dtype=x.dtype, device=x.device)

    def extra_repr(self):  # noqa: D401
        return f"C_in={self.C_in}, C_out={self.C_out}, stride={self.stride}"


# -----------------------------------------------------------------------------
#   Operator Factory
# -----------------------------------------------------------------------------

def _conv_creator(
    C_in: int,
    C_out: int,
    kernel_size: int,
    stride: int,
    padding: int,
) -> nn.Module:
    """Standard normal convolution."""

    return ReLUConvBN(C_in, C_out, kernel_size, stride, padding)


def _depthwise_creator(
    C_in: int,
    C_out: int,
    kernel_size: int,
    stride: int,
    padding: int,
) -> nn.Module:
    """Depth‑wise separable conv (single stack)."""

    return SepConv(C_in, C_out, kernel_size, stride, padding)


def _dual_depthwise_creator(
    C_in: int,
    C_out: int,
    kernel_size: int,
    stride: int,
    padding: int,
) -> nn.Module:
    """Depth‑wise separable conv (two‑stack)."""

    return DualSepConv(C_in, C_out, kernel_size, stride, padding)


def _group_conv_creator(
    C_in: int,
    C_out: int,
    kernel_size: int,
    stride: int,
    padding: int,
    groups: int = 4,
) -> nn.Module:
    return GroupConvBN(C_in, C_out, kernel_size, stride, padding, groups=groups)


# -----------------------------------------------------------------------------
#   Operator Mapping (energy‑aware)
# -----------------------------------------------------------------------------

OPS_Energy = {
    # Identity / None
    "none": lambda C_in, C_out=None, **kw: Zero(C_in, C_in if C_out is None else C_out, 1),
    "skip_connect": lambda C_in, C_out, stride=1, **kw: (
        Identity() if stride == 1 and C_in == C_out else Zero(C_in, C_out, stride)
    ),
    # Pooling
    "avg_pool_3x3": lambda C_in, C_out, stride=1, **kw: POOLING(
        C_in, C_out, stride, "avg", True, True, kernel_size=3
    ),
    "max_pool_3x3": lambda C_in, C_out, stride=1, **kw: POOLING(
        C_in, C_out, stride, "max", True, True, kernel_size=3
    ),
    "avg_pool_5x5": lambda C_in, C_out, stride=1, **kw: POOLING(
        C_in, C_out, stride, "avg", True, True, kernel_size=5
    ),
    "max_pool_5x5": lambda C_in, C_out, stride=1, **kw: POOLING(
        C_in, C_out, stride, "max", True, True, kernel_size=5
    ),
    # Normal convolutions (ReLU + Conv + BN)
    "nor_conv_1x1": lambda C_in, C_out, stride=1, **kw: _conv_creator(
        C_in, C_out, 1, stride, 0
    ),
    "nor_conv_3x3": lambda C_in, C_out, stride=1, **kw: _conv_creator(
        C_in, C_out, 3, stride, 1
    ),
    "nor_conv_5x5": lambda C_in, C_out, stride=1, **kw: _conv_creator(
        C_in, C_out, 5, stride, 2
    ),
    "nor_conv_7x7": lambda C_in, C_out, stride=1, **kw: _conv_creator(
        C_in, C_out, 7, stride, 3
    ),
    # Depth‑wise separable convolutions (single stack)
    "dw_conv_3x3": lambda C_in, C_out, stride=1, **kw: _depthwise_creator(
        C_in, C_out, 3, stride, 1
    ),
    "dw_conv_5x5": lambda C_in, C_out, stride=1, **kw: _depthwise_creator(
        C_in, C_out, 5, stride, 2
    ),
    "dw_conv_7x7": lambda C_in, C_out, stride=1, **kw: _depthwise_creator(
        C_in, C_out, 7, stride, 3
    ),
    # Dual depth‑wise separable conv (like MobileNetV2)
    "dua_sepc_3x3": lambda C_in, C_out, stride=1, **kw: _dual_depthwise_creator(
        C_in, C_out, 3, stride, 1
    ),
    "dua_sepc_5x5": lambda C_in, C_out, stride=1, **kw: _dual_depthwise_creator(
        C_in, C_out, 5, stride, 2
    ),
    # Dilated depth‑wise separable conv
    "dil_sepc_3x3": lambda C_in, C_out, stride=1, **kw: SepConv(
        C_in, C_out, 3, stride, 2, dilation=2
    ),
    "dil_sepc_5x5": lambda C_in, C_out, stride=1, **kw: SepConv(
        C_in, C_out, 5, stride, 4, dilation=2
    ),
    # Group conv
    "group_conv_3x3": lambda C_in, C_out, stride=1, **kw: _group_conv_creator(
        C_in, C_out, 3, stride, 1, groups=4
    ),
    # Shift (zero‑FLOPs)
    "shift": lambda C_in, C_out=None, **kw: Shift(C_in),
}

# Expose public alias (used by external NAS code).
OPS: Dict[str, Callable] = OPS_Energy

class ResNetBasicblockEnergy(nn.Module):
    def __init__(self, inplanes: int, planes: int, stride: int, affine: bool = True):
        super().__init__()
        assert stride in {1, 2}, f"invalid stride {stride}"
        self.conv_a = ReLUConvBN(inplanes, planes, 3, stride, 1, 1, affine)
        self.conv_b = ReLUConvBN(planes, planes, 3, 1, 1, 1, affine)
        if stride == 2:
            self.downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Conv2d(inplanes, planes, kernel_size=1, bias=False),
            )
        elif inplanes != planes:
            self.downsample = ReLUConvBN(inplanes, planes, 1, 1, 0, 1, affine)
        else:
            self.downsample = None
        self.in_dim = inplanes
        self.out_dim = planes
        self.stride = stride
        self.num_conv = 2

    def extra_repr(self):  # noqa: D401
        return (
            f"{self.__class__.__name__}(inC={self.in_dim}, "
            f"outC={self.out_dim}, stride={self.stride})"
        )

    def forward(self, x):  # noqa: D401
        out = self.conv_a(x)
        out = self.conv_b(out)
        residual = x if self.downsample is None else self.downsample(x)
        return residual + out

# -----------------------------------------------------------------------------
#   Search Space Definitions (for convenience)
# -----------------------------------------------------------------------------

CONNECT_NAS_BENCHMARK = ["none", "skip_connect", "nor_conv_3x3"]
NAS_BENCH_201 = [
    "none",
    "skip_connect",
    "nor_conv_1x1",
    "nor_conv_3x3",
    "avg_pool_3x3",
]
DARTS_SPACE = [
    "none",
    "skip_connect",
    "nor_conv_3x3",
    "nor_conv_5x5",
    "dua_sepc_3x3",
    "dua_sepc_5x5",
    "dil_sepc_3x3",
    "dil_sepc_5x5",
    "dw_conv_3x3",
    "avg_pool_3x3",
    "max_pool_3x3",
]

ENERGY_EFFICIENT_SPACE = [
    "none", 
    "skip_connect", 
    "nor_conv_1x1",
    "nor_conv_3x3",
    "nor_conv_5x5",
    "nor_conv_7x7",
    "dw_conv_3x3", 
    "dw_conv_5x5",
    "dw_conv_7x7",
    "avg_pool_3x3", 
    "avg_pool_5x5",
    "shift",
]

SearchSpaceNames = {
    "connect-nas": CONNECT_NAS_BENCHMARK,
    "nas-bench-201": NAS_BENCH_201,
    "darts": DARTS_SPACE,
    "energy": ENERGY_EFFICIENT_SPACE,         
    "energy-efficient": ENERGY_EFFICIENT_SPACE
}

