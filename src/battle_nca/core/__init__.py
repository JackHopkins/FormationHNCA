"""Core NCA components."""

from battle_nca.core.perceive import perceive, DepthwiseConvPerceive
from battle_nca.core.update import NCAUpdateRule
from battle_nca.core.nca import NCA

__all__ = ["perceive", "DepthwiseConvPerceive", "NCAUpdateRule", "NCA"]
