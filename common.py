"""Small convolution helper used by the paper-derived feature modules."""
from typing import Optional, Union

from torch import Tensor, nn


class ConvBNAct(nn.Module):
    """Convolution, batch normalization and an explicitly selectable activation.

    ``act=True`` uses SiLU, ``False`` is linear after BN, and an ``nn.Module``
    supplies another activation (e.g. the paper's CBR uses ``nn.ReLU``).
    This helper does not import or change an installed Ultralytics package.
    """

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1,
                 p: Optional[int] = None, g: int = 1,
                 act: Union[bool, nn.Module] = True):
        super().__init__()
        if c1 <= 0 or c2 <= 0 or k <= 0 or s <= 0 or g <= 0:
            raise ValueError("Channels, kernel, stride and groups must be positive.")
        if c1 % g or c2 % g:
            raise ValueError("Both channel counts must be divisible by groups.")
        if not isinstance(act, (bool, nn.Module)):
            raise TypeError("act must be a bool or an nn.Module instance.")
        self.conv = nn.Conv2d(c1, c2, k, s, k // 2 if p is None else p,
                              groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = act if isinstance(act, nn.Module) else (
            nn.SiLU(inplace=True) if act else nn.Identity())

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x)))
