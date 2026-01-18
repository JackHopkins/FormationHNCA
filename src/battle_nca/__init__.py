"""Battle simulator using Hierarchical Neural Cellular Automata."""

from battle_nca.core import NCA, perceive, DepthwiseConvPerceive, NCAUpdateRule
from battle_nca.hierarchy import ChildNCA, ParentNCA, HierarchicalNCA
from battle_nca.combat import Channels, FormationTargets, combat_loss, morale_loss
from battle_nca.training import NCAPool, Trainer

__version__ = "0.1.0"
__all__ = [
    "NCA",
    "perceive",
    "DepthwiseConvPerceive",
    "NCAUpdateRule",
    "ChildNCA",
    "ParentNCA",
    "HierarchicalNCA",
    "Channels",
    "FormationTargets",
    "combat_loss",
    "morale_loss",
    "NCAPool",
    "Trainer",
]
