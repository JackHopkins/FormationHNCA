"""Loss functions for battle NCA training."""

import jax
import jax.numpy as jnp
from flax import linen as nn

from battle_nca.combat.channels import Channels, MORALE, FLANKING


# Default channel configuration
CH = Channels()


def combat_loss(
    state_t0: jnp.ndarray,
    state_t1: jnp.ndarray,
    enemy_state: jnp.ndarray,
    damage_rate: float = 0.02
) -> jnp.ndarray:
    """Compute combat loss encouraging health decay when engaged.

    Trains the NCA to decrease health when adjacent to enemies.

    Args:
        state_t0: State before update
        state_t1: State after update
        enemy_state: Enemy army state
        damage_rate: Expected damage per step when engaged

    Returns:
        Combat loss scalar
    """
    # Detect enemy presence in neighborhood via max pooling
    enemy_alpha = enemy_state[..., CH.ALPHA:CH.ALPHA + 1]

    has_batch = enemy_alpha.ndim == 4
    if has_batch:
        window = (1, 3, 3, 1)
        strides = (1, 1, 1, 1)
    else:
        window = (3, 3, 1)
        strides = (1, 1, 1)

    enemy_nearby = jax.lax.reduce_window(
        enemy_alpha, -jnp.inf, jax.lax.max, window, strides, 'SAME'
    )[..., 0]

    # Cells in combat: both self and enemy present
    in_combat = (state_t0[..., CH.ALPHA] > 0.1) & (enemy_nearby > 0.1)

    # Expected health change
    expected_damage = in_combat.astype(jnp.float32) * damage_rate

    # Actual health change
    actual_change = state_t1[..., CH.HEALTH] - state_t0[..., CH.HEALTH]

    # Loss: encourage actual change to match expected (negative for damage)
    combat_error = (actual_change + expected_damage) ** 2

    # Only count alive cells
    alive_mask = state_t0[..., CH.ALPHA] > 0.1

    return jnp.sum(combat_error * alive_mask) / (jnp.sum(alive_mask) + 1e-6)


def morale_loss(
    state_t0: jnp.ndarray,
    state_t1: jnp.ndarray,
    routing_threshold: float = -0.5
) -> jnp.ndarray:
    """Compute morale propagation loss.

    Trains the NCA to decrease morale when surrounded by routing units
    (cascade routing effect from Total War).

    Args:
        state_t0: State before update
        state_t1: State after update
        routing_threshold: Morale value below which units are "routing"

    Returns:
        Morale loss scalar
    """
    morale_t0 = state_t0[..., CH.MORALE]
    morale_t1 = state_t1[..., CH.MORALE]

    # Count routing neighbors using average pooling
    routing_mask = (morale_t0 < routing_threshold).astype(jnp.float32)
    routing_mask = routing_mask[..., None]  # Add channel dim

    has_batch = routing_mask.ndim == 4
    if has_batch:
        window = (1, 5, 5, 1)
        strides = (1, 1, 1, 1)
    else:
        window = (5, 5, 1)
        strides = (1, 1, 1)

    routing_neighbors = jax.lax.reduce_window(
        routing_mask, 0.0, jax.lax.add, window, strides, 'SAME'
    )[..., 0]

    # Expected morale drop based on routing neighbors
    # Total War uses -12 leadership per 2 routing units
    expected_drop = routing_neighbors * MORALE.ROUTING_NEIGHBOR_PENALTY

    # Actual morale change
    actual_change = morale_t1 - morale_t0

    # Loss: encourage morale to decrease proportionally
    morale_error = (actual_change - expected_drop) ** 2

    # Only count alive, non-routing cells
    alive_mask = (state_t0[..., CH.ALPHA] > 0.1) & (morale_t0 > routing_threshold)

    return jnp.sum(morale_error * alive_mask) / (jnp.sum(alive_mask) + 1e-6)


def formation_loss(
    state: jnp.ndarray,
    target: jnp.ndarray
) -> jnp.ndarray:
    """Compute formation fidelity loss (MSE on RGBA channels).

    Args:
        state: Current state
        target: Target formation (RGBA)

    Returns:
        Formation loss scalar
    """
    state_rgba = state[..., :4]
    target_rgba = target[..., :4] if target.shape[-1] >= 4 else target

    # Ensure shapes match
    if state_rgba.shape != target_rgba.shape:
        # Handle batch dimension
        if state_rgba.ndim == 4 and target_rgba.ndim == 3:
            target_rgba = target_rgba[None]
            target_rgba = jnp.broadcast_to(
                target_rgba, state_rgba.shape
            )

    return jnp.mean((state_rgba - target_rgba) ** 2)


def overflow_loss(
    state: jnp.ndarray,
    min_val: float = -1.0,
    max_val: float = 2.0
) -> jnp.ndarray:
    """Auxiliary loss to prevent state explosion.

    Penalizes values outside [min_val, max_val] range.

    Args:
        state: Current state
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Returns:
        Overflow loss scalar
    """
    overflow = jax.nn.relu(state - max_val) + jax.nn.relu(min_val - state)
    return jnp.mean(overflow)


def velocity_coherence_loss(
    state: jnp.ndarray,
    neighbor_radius: int = 3
) -> jnp.ndarray:
    """Loss encouraging local velocity alignment (flocking behavior).

    Args:
        state: Current state
        neighbor_radius: Radius for neighbor averaging

    Returns:
        Velocity coherence loss
    """
    vx = state[..., CH.VELOCITY_X:CH.VELOCITY_X + 1]
    vy = state[..., CH.VELOCITY_Y:CH.VELOCITY_Y + 1]

    has_batch = vx.ndim == 4
    size = 2 * neighbor_radius + 1

    if has_batch:
        window = (1, size, size, 1)
        strides = (1, 1, 1, 1)
    else:
        window = (size, size, 1)
        strides = (1, 1, 1)

    # Average neighbor velocity
    avg_vx = jax.lax.reduce_window(
        vx, 0.0, jax.lax.add, window, strides, 'SAME'
    ) / (size * size)
    avg_vy = jax.lax.reduce_window(
        vy, 0.0, jax.lax.add, window, strides, 'SAME'
    ) / (size * size)

    # Deviation from local average
    vx_diff = (vx - avg_vx) ** 2
    vy_diff = (vy - avg_vy) ** 2

    # Only count alive cells
    alive_mask = state[..., CH.ALPHA:CH.ALPHA + 1] > 0.1

    return jnp.mean((vx_diff + vy_diff) * alive_mask)


def regeneration_loss(
    state_damaged: jnp.ndarray,
    state_healed: jnp.ndarray,
    target: jnp.ndarray,
    num_heal_steps: int = 50
) -> jnp.ndarray:
    """Loss for regeneration/reformation capability.

    Encourages the NCA to recover target formation after damage.

    Args:
        state_damaged: State immediately after damage
        state_healed: State after healing steps
        target: Target formation
        num_heal_steps: Number of steps taken to heal

    Returns:
        Regeneration loss
    """
    # After healing, should approach target
    return formation_loss(state_healed, target)


def casualty_ratio_loss(
    red_state: jnp.ndarray,
    blue_state: jnp.ndarray,
    target_ratio: float = 1.0
) -> jnp.ndarray:
    """Loss encouraging favorable casualty ratio.

    Args:
        red_state: Red army state
        blue_state: Blue army state
        target_ratio: Target red/blue casualty ratio (>1 = red winning)

    Returns:
        Casualty ratio loss
    """
    red_alive = jnp.sum(red_state[..., CH.ALPHA] > 0.1)
    blue_alive = jnp.sum(blue_state[..., CH.ALPHA] > 0.1)

    # Avoid division by zero
    actual_ratio = red_alive / (blue_alive + 1e-6)

    return (actual_ratio - target_ratio) ** 2


def formation_integrity_loss(state: jnp.ndarray) -> jnp.ndarray:
    """Loss encouraging formation cohesion.

    Penalizes isolated units (units without nearby allies).

    Args:
        state: Army state

    Returns:
        Formation integrity loss
    """
    alpha = state[..., CH.ALPHA:CH.ALPHA + 1]

    has_batch = alpha.ndim == 4
    if has_batch:
        window = (1, 5, 5, 1)
        strides = (1, 1, 1, 1)
    else:
        window = (5, 5, 1)
        strides = (1, 1, 1)

    # Count nearby allies
    ally_count = jax.lax.reduce_window(
        alpha, 0.0, jax.lax.add, window, strides, 'SAME'
    )

    # Cells with few allies are isolated
    isolated = (alpha > 0.1) & (ally_count < 3)

    return jnp.mean(isolated.astype(jnp.float32))


def total_battle_loss(
    state_t0: jnp.ndarray,
    state_t1: jnp.ndarray,
    target: jnp.ndarray,
    enemy_state: jnp.ndarray | None = None,
    weights: dict[str, float] | None = None
) -> dict[str, jnp.ndarray]:
    """Compute total battle loss with all components.

    Args:
        state_t0: State before update
        state_t1: State after update
        target: Target formation
        enemy_state: Optional enemy state for combat loss
        weights: Optional loss weights

    Returns:
        Dictionary with individual and total losses
    """
    if weights is None:
        weights = {
            'formation': 1.0,
            'combat': 0.5,
            'morale': 0.3,
            'overflow': 0.1,
            'velocity': 0.1,
            'integrity': 0.2
        }

    losses = {}

    # Formation loss
    losses['formation'] = formation_loss(state_t1, target)

    # Combat loss (if enemy present)
    if enemy_state is not None:
        losses['combat'] = combat_loss(state_t0, state_t1, enemy_state)
    else:
        losses['combat'] = jnp.array(0.0)

    # Morale loss
    losses['morale'] = morale_loss(state_t0, state_t1)

    # Overflow loss
    losses['overflow'] = overflow_loss(state_t1)

    # Velocity coherence
    losses['velocity'] = velocity_coherence_loss(state_t1)

    # Formation integrity
    losses['integrity'] = formation_integrity_loss(state_t1)

    # Total weighted loss
    total = sum(weights.get(k, 0.0) * v for k, v in losses.items())
    losses['total'] = total

    return losses
