"""Formation targets and utilities for battle simulation."""

from enum import IntEnum
from dataclasses import dataclass
import jax.numpy as jnp


class FormationTypes(IntEnum):
    """Available formation types."""
    LINE = 0
    PHALANX = 1
    SQUARE = 2
    WEDGE = 3
    COLUMN = 4
    TESTUDO = 5


@dataclass
class FormationSpec:
    """Specification for a formation type.

    Attributes:
        name: Formation name
        depth: Number of ranks
        spacing: Unit spacing (1.0 = shoulder to shoulder)
        density: Target unit density
        description: Formation description
    """
    name: str
    depth: int
    spacing: float
    density: float
    description: str


# Formation specifications based on historical/Total War data
FORMATION_SPECS: dict[FormationTypes, FormationSpec] = {
    FormationTypes.LINE: FormationSpec(
        name="Line",
        depth=2,
        spacing=0.8,
        density=0.85,
        description="Thin line, maximum frontage"
    ),
    FormationTypes.PHALANX: FormationSpec(
        name="Phalanx",
        depth=16,
        spacing=0.95,
        density=0.95,
        description="Deep formation, 16 ranks for push power"
    ),
    FormationTypes.SQUARE: FormationSpec(
        name="Square",
        depth=4,
        spacing=0.9,
        density=0.90,
        description="Hollow defensive formation"
    ),
    FormationTypes.WEDGE: FormationSpec(
        name="Wedge",
        depth=8,
        spacing=0.85,
        density=0.80,
        description="Triangle for breaking enemy lines"
    ),
    FormationTypes.COLUMN: FormationSpec(
        name="Column",
        depth=20,
        spacing=0.7,
        density=0.75,
        description="Narrow deep column for marching"
    ),
    FormationTypes.TESTUDO: FormationSpec(
        name="Testudo",
        depth=5,
        spacing=1.0,
        density=1.0,
        description="Maximum density defensive formation"
    ),
}


class FormationTargets:
    """Factory for creating formation target patterns."""

    @staticmethod
    def line(height: int, width: int, depth: int = 2) -> jnp.ndarray:
        """Create line formation target.

        Args:
            height: Grid height
            width: Grid width
            depth: Number of ranks (default 2)

        Returns:
            RGBA target tensor
        """
        target = jnp.zeros((height, width, 4))
        center_row = height // 2
        start_row = center_row - depth // 2
        end_row = start_row + depth

        target = target.at[start_row:end_row, :, 3].set(
            FORMATION_SPECS[FormationTypes.LINE].density
        )
        target = target.at[start_row:end_row, :, :3].set(1.0)

        return target

    @staticmethod
    def phalanx(height: int, width: int, depth: int = 16) -> jnp.ndarray:
        """Create phalanx formation target.

        Args:
            height: Grid height
            width: Grid width
            depth: Number of ranks (default 16)

        Returns:
            RGBA target tensor
        """
        target = jnp.zeros((height, width, 4))
        depth = min(depth, height - 4)
        center_row = height // 2
        start_row = center_row - depth // 2
        end_row = start_row + depth

        density = FORMATION_SPECS[FormationTypes.PHALANX].density
        target = target.at[start_row:end_row, :, 3].set(density)
        target = target.at[start_row:end_row, :, :3].set(1.0)

        return target

    @staticmethod
    def square(height: int, width: int, thickness: int = 4) -> jnp.ndarray:
        """Create hollow square formation target.

        Args:
            height: Grid height
            width: Grid width
            thickness: Wall thickness (default 4)

        Returns:
            RGBA target tensor
        """
        target = jnp.zeros((height, width, 4))
        density = FORMATION_SPECS[FormationTypes.SQUARE].density

        # Top wall
        target = target.at[:thickness, :, 3].set(density)
        # Bottom wall
        target = target.at[-thickness:, :, 3].set(density)
        # Left wall
        target = target.at[:, :thickness, 3].set(density)
        # Right wall
        target = target.at[:, -thickness:, 3].set(density)

        # Set RGB where alpha > 0
        target = target.at[..., :3].set(
            jnp.where(target[..., 3:4] > 0, 1.0, 0.0)
        )

        return target

    @staticmethod
    def wedge(height: int, width: int) -> jnp.ndarray:
        """Create wedge (triangle) formation target.

        Args:
            height: Grid height
            width: Grid width

        Returns:
            RGBA target tensor
        """
        target = jnp.zeros((height, width, 4))
        density = FORMATION_SPECS[FormationTypes.WEDGE].density

        # Create triangular density gradient
        for row in range(height):
            progress = row / height
            half_width = int((1 - progress) * width / 2)
            center = width // 2

            if half_width > 0:
                row_density = density * (0.5 + 0.5 * progress)
                target = target.at[row, center - half_width:center + half_width, 3].set(
                    row_density
                )

        target = target.at[..., :3].set(
            jnp.where(target[..., 3:4] > 0, 1.0, 0.0)
        )

        return target

    @staticmethod
    def column(height: int, width: int, col_width: int = 4) -> jnp.ndarray:
        """Create column formation target.

        Args:
            height: Grid height
            width: Grid width
            col_width: Column width (default 4)

        Returns:
            RGBA target tensor
        """
        target = jnp.zeros((height, width, 4))
        density = FORMATION_SPECS[FormationTypes.COLUMN].density
        center = width // 2
        half = col_width // 2

        target = target.at[:, center - half:center + half, 3].set(density)
        target = target.at[:, center - half:center + half, :3].set(1.0)

        return target

    @staticmethod
    def testudo(height: int, width: int, size: int = 10) -> jnp.ndarray:
        """Create testudo formation target.

        Args:
            height: Grid height
            width: Grid width
            size: Formation size (default 10x10)

        Returns:
            RGBA target tensor
        """
        target = jnp.zeros((height, width, 4))
        density = FORMATION_SPECS[FormationTypes.TESTUDO].density

        center_y, center_x = height // 2, width // 2
        half = size // 2

        target = target.at[
            center_y - half:center_y + half,
            center_x - half:center_x + half,
            3
        ].set(density)
        target = target.at[
            center_y - half:center_y + half,
            center_x - half:center_x + half,
            :3
        ].set(1.0)

        return target


def create_formation_target(
    height: int,
    width: int,
    formation_type: FormationTypes | int | str
) -> jnp.ndarray:
    """Create formation target pattern.

    Args:
        height: Grid height
        width: Grid width
        formation_type: Formation type (enum, int, or string name)

    Returns:
        RGBA target tensor
    """
    # Convert to FormationTypes if needed
    if isinstance(formation_type, str):
        formation_type = FormationTypes[formation_type.upper()]
    elif isinstance(formation_type, int):
        formation_type = FormationTypes(formation_type)

    targets = FormationTargets()

    if formation_type == FormationTypes.LINE:
        return targets.line(height, width)
    elif formation_type == FormationTypes.PHALANX:
        return targets.phalanx(height, width)
    elif formation_type == FormationTypes.SQUARE:
        return targets.square(height, width)
    elif formation_type == FormationTypes.WEDGE:
        return targets.wedge(height, width)
    elif formation_type == FormationTypes.COLUMN:
        return targets.column(height, width)
    elif formation_type == FormationTypes.TESTUDO:
        return targets.testudo(height, width)
    else:
        raise ValueError(f"Unknown formation type: {formation_type}")


def create_all_formation_targets(
    height: int,
    width: int
) -> dict[FormationTypes, jnp.ndarray]:
    """Create all formation targets.

    Args:
        height: Grid height
        width: Grid width

    Returns:
        Dictionary mapping FormationTypes to RGBA targets
    """
    return {
        ft: create_formation_target(height, width, ft)
        for ft in FormationTypes
    }


def measure_formation_quality(
    state: jnp.ndarray,
    target: jnp.ndarray,
    alpha_channel: int = 3
) -> dict[str, float]:
    """Measure how well state matches formation target.

    Args:
        state: Current NCA state
        target: Target formation
        alpha_channel: Index of alpha channel

    Returns:
        Dictionary with quality metrics
    """
    state_alpha = state[..., alpha_channel]
    target_alpha = target[..., alpha_channel]

    # MSE
    mse = jnp.mean((state_alpha - target_alpha) ** 2)

    # IoU (Intersection over Union)
    threshold = 0.5
    state_mask = state_alpha > threshold
    target_mask = target_alpha > threshold
    intersection = jnp.sum(state_mask & target_mask)
    union = jnp.sum(state_mask | target_mask)
    iou = intersection / (union + 1e-6)

    # Coverage (what fraction of target is filled)
    coverage = jnp.sum(state_mask & target_mask) / (jnp.sum(target_mask) + 1e-6)

    # Precision (what fraction of state is in target)
    precision = jnp.sum(state_mask & target_mask) / (jnp.sum(state_mask) + 1e-6)

    return {
        'mse': float(mse),
        'iou': float(iou),
        'coverage': float(coverage),
        'precision': float(precision)
    }
