"""Paper-based reference implementations of the two enhancement modules."""
from .c3k2_gf import C3k2_GF, C3k2GF, C3k_GF, GF_Bottleneck, GFCore, TNSG
from .td_cfem import (
    DEFAULT_FREQUENCIES,
    EAFS,
    ECA,
    LaplacianSpatialGate,
    PFD,
    PFSG,
    TD_CFEM,
    TDCFEM,
)

__all__ = [
    "C3k2_GF", "C3k2GF", "C3k_GF", "GF_Bottleneck", "GFCore", "TNSG",
    "DEFAULT_FREQUENCIES", "EAFS", "ECA", "LaplacianSpatialGate", "PFD",
    "PFSG", "TD_CFEM", "TDCFEM",
]
