"""Formation targets and utilities for battle simulation."""

from enum import IntEnum
from dataclasses import dataclass
import jax
import jax.numpy as jnp
from functools import partial


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


def rotate_formation(
    target: jnp.ndarray,
    angle: float,
    order: int = 1
) -> jnp.ndarray:
    """Rotate a formation target by the given angle.

    Uses bilinear interpolation for smooth rotation.

    Args:
        target: Formation target of shape (H, W, C)
        angle: Rotation angle in radians (counterclockwise)
        order: Interpolation order (1 for bilinear)

    Returns:
        Rotated formation target
    """
    h, w, c = target.shape
    center_y, center_x = h / 2, w / 2

    # Create coordinate grids for output
    y_out, x_out = jnp.meshgrid(
        jnp.arange(h, dtype=jnp.float32),
        jnp.arange(w, dtype=jnp.float32),
        indexing='ij'
    )

    # Translate to center
    y_centered = y_out - center_y
    x_centered = x_out - center_x

    # Apply inverse rotation (to find source coordinates)
    cos_a = jnp.cos(angle)
    sin_a = jnp.sin(angle)
    y_src = cos_a * y_centered + sin_a * x_centered + center_y
    x_src = -sin_a * y_centered + cos_a * x_centered + center_x

    # Bilinear interpolation
    y0 = jnp.floor(y_src).astype(jnp.int32)
    x0 = jnp.floor(x_src).astype(jnp.int32)
    y1 = y0 + 1
    x1 = x0 + 1

    # Clamp to valid range
    y0_c = jnp.clip(y0, 0, h - 1)
    y1_c = jnp.clip(y1, 0, h - 1)
    x0_c = jnp.clip(x0, 0, w - 1)
    x1_c = jnp.clip(x1, 0, w - 1)

    # Interpolation weights
    wy = y_src - y0.astype(jnp.float32)
    wx = x_src - x0.astype(jnp.float32)

    # Out of bounds mask (set to zero)
    in_bounds = (
        (y_src >= 0) & (y_src < h - 1) &
        (x_src >= 0) & (x_src < w - 1)
    )

    # Gather values at corners and interpolate
    def interpolate_channel(channel_data):
        v00 = channel_data[y0_c, x0_c]
        v01 = channel_data[y0_c, x1_c]
        v10 = channel_data[y1_c, x0_c]
        v11 = channel_data[y1_c, x1_c]

        # Bilinear interpolation
        v0 = v00 * (1 - wx) + v01 * wx
        v1 = v10 * (1 - wx) + v11 * wx
        v = v0 * (1 - wy) + v1 * wy

        # Zero outside bounds
        return jnp.where(in_bounds, v, 0.0)

    # Apply to all channels
    rotated = jnp.stack([
        interpolate_channel(target[..., i])
        for i in range(c)
    ], axis=-1)

    return rotated


def create_rotated_variants(
    target: jnp.ndarray,
    num_rotations: int = 8
) -> list[jnp.ndarray]:
    """Create rotated variants of a formation target.

    Args:
        target: Base formation target
        num_rotations: Number of rotation variants (evenly spaced)

    Returns:
        List of rotated formation targets
    """
    angles = jnp.linspace(0, 2 * jnp.pi, num_rotations, endpoint=False)
    return [rotate_formation(target, float(angle)) for angle in angles]


@partial(jax.jit, static_argnums=(1, 2))
def random_rotate_formation(
    target: jnp.ndarray,
    key: jax.random.PRNGKey,
    continuous: bool = True
) -> jnp.ndarray:
    """Randomly rotate a formation target.

    Args:
        target: Formation target
        key: PRNG key
        continuous: If True, use continuous angles; if False, use 90° increments

    Returns:
        Randomly rotated formation target
    """
    if continuous:
        angle = jax.random.uniform(key, (), minval=0, maxval=2 * jnp.pi)
    else:
        # 90 degree increments
        idx = jax.random.randint(key, (), 0, 4)
        angle = idx * (jnp.pi / 2)

    return rotate_formation(target, angle)


# Scale factor to ensure formations fit when rotated 45°
# diagonal = side * sqrt(2), so we need side = grid / sqrt(2) ≈ 0.707 * grid
ROTATION_SAFE_SCALE = 1.0 / jnp.sqrt(2.0)  # ~0.707


class FormationTargets:
    """Factory for creating formation target patterns."""

    @staticmethod
    def line(height: int, width: int, depth: int = 2, rotation_safe: bool = False) -> jnp.ndarray:
        """Create line formation target.

        Args:
            height: Grid height
            width: Grid width
            depth: Number of ranks (default 2)
            rotation_safe: If True, scale formation so diagonal fits in grid

        Returns:
            RGBA target tensor
        """
        target = jnp.zeros((height, width, 4))
        center_row = height // 2
        center_col = width // 2

        if rotation_safe:
            # Scale width so diagonal fits: effective_width = width * 0.707
            margin = int(width * (1 - ROTATION_SAFE_SCALE) / 2)
            start_col = margin
            end_col = width - margin
        else:
            start_col = 0
            end_col = width

        start_row = center_row - depth // 2
        end_row = start_row + depth

        target = target.at[start_row:end_row, start_col:end_col, 3].set(
            FORMATION_SPECS[FormationTypes.LINE].density
        )
        target = target.at[start_row:end_row, start_col:end_col, :3].set(1.0)

        return target

    @staticmethod
    def phalanx(height: int, width: int, depth: int = 16, rotation_safe: bool = False) -> jnp.ndarray:
        """Create phalanx formation target.

        Args:
            height: Grid height
            width: Grid width
            depth: Number of ranks (default 16)
            rotation_safe: If True, scale formation so diagonal fits in grid

        Returns:
            RGBA target tensor
        """
        target = jnp.zeros((height, width, 4))

        if rotation_safe:
            # Scale both dimensions
            h_margin = int(height * (1 - ROTATION_SAFE_SCALE) / 2)
            w_margin = int(width * (1 - ROTATION_SAFE_SCALE) / 2)
            effective_height = height - 2 * h_margin
            depth = min(depth, effective_height - 4)
            start_col = w_margin
            end_col = width - w_margin
        else:
            depth = min(depth, height - 4)
            start_col = 0
            end_col = width

        center_row = height // 2
        start_row = center_row - depth // 2
        end_row = start_row + depth

        density = FORMATION_SPECS[FormationTypes.PHALANX].density
        target = target.at[start_row:end_row, start_col:end_col, 3].set(density)
        target = target.at[start_row:end_row, start_col:end_col, :3].set(1.0)

        return target

    @staticmethod
    def square(height: int, width: int, thickness: int = 4, rotation_safe: bool = False) -> jnp.ndarray:
        """Create hollow square formation target.

        Args:
            height: Grid height
            width: Grid width
            thickness: Wall thickness (default 4)
            rotation_safe: If True, scale formation so diagonal fits in grid

        Returns:
            RGBA target tensor
        """
        target = jnp.zeros((height, width, 4))
        density = FORMATION_SPECS[FormationTypes.SQUARE].density

        if rotation_safe:
            # Scale to fit when rotated: margin = (1 - 0.707) / 2 ≈ 0.146
            h_margin = int(height * (1 - ROTATION_SAFE_SCALE) / 2)
            w_margin = int(width * (1 - ROTATION_SAFE_SCALE) / 2)
            top = h_margin
            bottom = height - h_margin
            left = w_margin
            right = width - w_margin
        else:
            top = 0
            bottom = height
            left = 0
            right = width

        # Top wall
        target = target.at[top:top+thickness, left:right, 3].set(density)
        # Bottom wall
        target = target.at[bottom-thickness:bottom, left:right, 3].set(density)
        # Left wall
        target = target.at[top:bottom, left:left+thickness, 3].set(density)
        # Right wall
        target = target.at[top:bottom, right-thickness:right, 3].set(density)

        # Set RGB where alpha > 0
        target = target.at[..., :3].set(
            jnp.where(target[..., 3:4] > 0, 1.0, 0.0)
        )

        return target

    @staticmethod
    def wedge(height: int, width: int, rotation_safe: bool = False) -> jnp.ndarray:
        """Create wedge (triangle) formation target.

        Args:
            height: Grid height
            width: Grid width
            rotation_safe: If True, scale formation so diagonal fits in grid

        Returns:
            RGBA target tensor
        """
        target = jnp.zeros((height, width, 4))
        density = FORMATION_SPECS[FormationTypes.WEDGE].density

        if rotation_safe:
            h_margin = int(height * (1 - ROTATION_SAFE_SCALE) / 2)
            w_margin = int(width * (1 - ROTATION_SAFE_SCALE) / 2)
            effective_height = height - 2 * h_margin
            effective_width = width - 2 * w_margin
            start_row = h_margin
        else:
            h_margin = 0
            w_margin = 0
            effective_height = height
            effective_width = width
            start_row = 0

        center = width // 2

        # Create triangular density gradient
        for i in range(effective_height):
            row = start_row + i
            progress = i / effective_height
            half_width = int((1 - progress) * effective_width / 2)

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
    def column(height: int, width: int, col_width: int = 4, rotation_safe: bool = False) -> jnp.ndarray:
        """Create column formation target.

        Args:
            height: Grid height
            width: Grid width
            col_width: Column width (default 4)
            rotation_safe: If True, scale formation so diagonal fits in grid

        Returns:
            RGBA target tensor
        """
        target = jnp.zeros((height, width, 4))
        density = FORMATION_SPECS[FormationTypes.COLUMN].density
        center = width // 2
        half = col_width // 2

        if rotation_safe:
            h_margin = int(height * (1 - ROTATION_SAFE_SCALE) / 2)
            start_row = h_margin
            end_row = height - h_margin
        else:
            start_row = 0
            end_row = height

        target = target.at[start_row:end_row, center - half:center + half, 3].set(density)
        target = target.at[start_row:end_row, center - half:center + half, :3].set(1.0)

        return target

    @staticmethod
    def testudo(height: int, width: int, size: int = 10, rotation_safe: bool = False) -> jnp.ndarray:
        """Create testudo formation target.

        Args:
            height: Grid height
            width: Grid width
            size: Formation size (default 10x10)
            rotation_safe: If True, scale formation so diagonal fits in grid

        Returns:
            RGBA target tensor
        """
        target = jnp.zeros((height, width, 4))
        density = FORMATION_SPECS[FormationTypes.TESTUDO].density

        if rotation_safe:
            # Reduce size so diagonal fits
            size = int(size * ROTATION_SAFE_SCALE)

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
    formation_type: FormationTypes | int | str,
    rotation: float = 0.0,
    rotation_safe: bool = False
) -> jnp.ndarray:
    """Create formation target pattern.

    Args:
        height: Grid height
        width: Grid width
        formation_type: Formation type (enum, int, or string name)
        rotation: Rotation angle in radians (default 0)
        rotation_safe: If True, scale formation so diagonal fits when rotated

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
        target = targets.line(height, width, rotation_safe=rotation_safe)
    elif formation_type == FormationTypes.PHALANX:
        target = targets.phalanx(height, width, rotation_safe=rotation_safe)
    elif formation_type == FormationTypes.SQUARE:
        target = targets.square(height, width, rotation_safe=rotation_safe)
    elif formation_type == FormationTypes.WEDGE:
        target = targets.wedge(height, width, rotation_safe=rotation_safe)
    elif formation_type == FormationTypes.COLUMN:
        target = targets.column(height, width, rotation_safe=rotation_safe)
    elif formation_type == FormationTypes.TESTUDO:
        target = targets.testudo(height, width, rotation_safe=rotation_safe)
    else:
        raise ValueError(f"Unknown formation type: {formation_type}")

    # Apply rotation if specified
    if rotation != 0.0:
        target = rotate_formation(target, rotation)

    return target


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
