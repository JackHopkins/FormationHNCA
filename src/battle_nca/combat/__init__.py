"""Combat mechanics and loss functions."""

from battle_nca.combat.channels import Channels, UnitTypes
from battle_nca.combat.formations import FormationTargets, create_formation_target
from battle_nca.combat.losses import (
    combat_loss,
    morale_loss,
    formation_loss,
    overflow_loss,
    total_battle_loss
)

__all__ = [
    "Channels",
    "UnitTypes",
    "FormationTargets",
    "create_formation_target",
    "combat_loss",
    "morale_loss",
    "formation_loss",
    "overflow_loss",
    "total_battle_loss",
]
