"""Core NCA components."""

from battle_nca.core.perceive import perceive, DepthwiseConvPerceive
from battle_nca.core.update import NCAUpdateRule
from battle_nca.core.nca import NCA
from battle_nca.core.advection import advect_mass, advect_mass_circular, check_mass_conservation

__all__ = [
    "perceive", "DepthwiseConvPerceive", "NCAUpdateRule", "NCA",
    "advect_mass", "advect_mass_circular", "check_mass_conservation"
]
