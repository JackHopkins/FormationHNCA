"""Channel definitions and unit type encodings for battle simulation."""

from dataclasses import dataclass
from enum import IntEnum
import jax.numpy as jnp


@dataclass(frozen=True)
class Channels:
    """Channel allocation for battle NCA state.

    Child-NCA uses 24 channels to encode combat state, movement,
    coordination signals, and hidden state.
    """
    # Visualization (0-3)
    RGB_R: int = 0
    RGB_G: int = 1
    RGB_B: int = 2
    ALPHA: int = 3

    # Combat stats (4-6)
    HEALTH: int = 4
    MORALE: int = 5
    FATIGUE: int = 6

    # Movement (7-8)
    VELOCITY_X: int = 7
    VELOCITY_Y: int = 8

    # Identity (9-10)
    UNIT_TYPE: int = 9
    FORMATION_ID: int = 10

    # Communication (11-14)
    PARENT_SIGNAL_0: int = 11
    PARENT_SIGNAL_1: int = 12
    ENEMY_PROXIMITY: int = 13
    ENEMY_DIRECTION: int = 14

    # Hidden state (15-23)
    HIDDEN_START: int = 15
    HIDDEN_END: int = 24

    TOTAL: int = 24

    # Channel groups
    @property
    def rgb_slice(self) -> slice:
        return slice(0, 3)

    @property
    def rgba_slice(self) -> slice:
        return slice(0, 4)

    @property
    def combat_slice(self) -> slice:
        return slice(4, 7)

    @property
    def velocity_slice(self) -> slice:
        return slice(7, 9)

    @property
    def identity_slice(self) -> slice:
        return slice(9, 11)

    @property
    def parent_signal_slice(self) -> slice:
        return slice(11, 13)

    @property
    def enemy_info_slice(self) -> slice:
        return slice(13, 15)

    @property
    def hidden_slice(self) -> slice:
        return slice(15, 24)


class UnitTypes(IntEnum):
    """Unit type encodings.

    Values are stored as floats in channel 9, normalized to [0, 1].
    """
    INFANTRY = 0
    CAVALRY = 1
    ARCHER = 2
    PIKE = 3
    HEAVY_INFANTRY = 4

    @classmethod
    def encode(cls, unit_type: 'UnitTypes') -> float:
        """Encode unit type as normalized float."""
        return float(unit_type) / (len(cls) - 1)

    @classmethod
    def decode(cls, value: float) -> 'UnitTypes':
        """Decode normalized float to unit type."""
        idx = int(round(value * (len(cls) - 1)))
        return cls(idx)


@dataclass(frozen=True)
class UnitStats:
    """Combat statistics for each unit type.

    Based on Total War mechanics:
    - Attack: Damage dealt per combat tick
    - Defense: Damage reduction multiplier
    - Morale: Base morale value
    - Speed: Movement speed multiplier
    - Charge: Bonus damage when charging
    """
    attack: float
    defense: float
    morale: float
    speed: float
    charge: float


# Unit type statistics lookup
UNIT_STATS: dict[UnitTypes, UnitStats] = {
    UnitTypes.INFANTRY: UnitStats(
        attack=1.0, defense=0.5, morale=0.6, speed=1.0, charge=0.2
    ),
    UnitTypes.CAVALRY: UnitStats(
        attack=1.2, defense=0.3, morale=0.7, speed=2.0, charge=1.5
    ),
    UnitTypes.ARCHER: UnitStats(
        attack=0.6, defense=0.2, morale=0.4, speed=1.0, charge=0.0
    ),
    UnitTypes.PIKE: UnitStats(
        attack=0.8, defense=0.7, morale=0.5, speed=0.7, charge=-0.5  # Anti-charge
    ),
    UnitTypes.HEAVY_INFANTRY: UnitStats(
        attack=1.3, defense=0.8, morale=0.8, speed=0.6, charge=0.3
    ),
}


def get_unit_stats_tensor(unit_type_channel: jnp.ndarray) -> dict[str, jnp.ndarray]:
    """Convert unit type channel to stat tensors.

    Args:
        unit_type_channel: Tensor of encoded unit types (H, W) or (B, H, W)

    Returns:
        Dictionary with attack, defense, morale, speed, charge tensors
    """
    # Create lookup tables
    num_types = len(UnitTypes)
    attack_table = jnp.array([UNIT_STATS[UnitTypes(i)].attack for i in range(num_types)])
    defense_table = jnp.array([UNIT_STATS[UnitTypes(i)].defense for i in range(num_types)])
    morale_table = jnp.array([UNIT_STATS[UnitTypes(i)].morale for i in range(num_types)])
    speed_table = jnp.array([UNIT_STATS[UnitTypes(i)].speed for i in range(num_types)])
    charge_table = jnp.array([UNIT_STATS[UnitTypes(i)].charge for i in range(num_types)])

    # Convert normalized values to indices
    indices = jnp.round(unit_type_channel * (num_types - 1)).astype(jnp.int32)
    indices = jnp.clip(indices, 0, num_types - 1)

    return {
        'attack': attack_table[indices],
        'defense': defense_table[indices],
        'morale': morale_table[indices],
        'speed': speed_table[indices],
        'charge': charge_table[indices],
    }


# Flanking modifiers (from Total War)
@dataclass(frozen=True)
class FlankingModifiers:
    """Defense modifiers based on attack angle.

    Total War uses:
    - Front: 100% defense
    - Flank (side): 60% defense
    - Rear: 25% defense
    """
    FRONT: float = 1.0
    FLANK: float = 0.6
    REAR: float = 0.25

    @staticmethod
    def compute_modifier(
        facing_x: jnp.ndarray,
        facing_y: jnp.ndarray,
        attack_x: jnp.ndarray,
        attack_y: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute defense modifier based on attack angle.

        Args:
            facing_x, facing_y: Unit facing direction (normalized)
            attack_x, attack_y: Attack direction (normalized)

        Returns:
            Defense modifier in [0.25, 1.0]
        """
        # Dot product gives cos(angle)
        dot = facing_x * attack_x + facing_y * attack_y

        # Front: dot > 0.5 (within ~60 degrees)
        # Flank: -0.5 < dot < 0.5
        # Rear: dot < -0.5

        modifier = jnp.where(
            dot > 0.5,
            FlankingModifiers.FRONT,
            jnp.where(
                dot < -0.5,
                FlankingModifiers.REAR,
                FlankingModifiers.FLANK
            )
        )

        return modifier


FLANKING = FlankingModifiers()


# Morale modifiers
@dataclass(frozen=True)
class MoraleModifiers:
    """Morale impact values.

    Based on Total War leadership mechanics.
    """
    ROUTING_NEIGHBOR_PENALTY: float = -0.12  # Per routing neighbor
    CASUALTY_PENALTY: float = -0.05  # Per % casualties
    FLANKED_PENALTY: float = -0.15
    SURROUNDED_PENALTY: float = -0.25
    WINNING_BONUS: float = 0.10
    GENERAL_NEARBY_BONUS: float = 0.20


MORALE = MoraleModifiers()
