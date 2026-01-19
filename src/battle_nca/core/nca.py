"""Base Neural Cellular Automata module."""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Callable

from battle_nca.core.perceive import perceive, DepthwiseConvPerceive
from battle_nca.core.update import NCAUpdateRule


def stochastic_update(
    state: jnp.ndarray,
    ds: jnp.ndarray,
    key: jax.random.PRNGKey,
    fire_rate: float = 0.5
) -> jnp.ndarray:
    """Apply stochastic cell update mask.

    Args:
        state: Current state of shape (H, W, C) or (B, H, W, C)
        ds: Residual update of same shape
        key: PRNG key for stochastic masking
        fire_rate: Probability of each cell updating

    Returns:
        Updated state with stochastic mask applied
    """
    has_batch = state.ndim == 4
    if has_batch:
        shape = state.shape[:3]  # (B, H, W)
    else:
        shape = state.shape[:2]  # (H, W)

    mask = jax.random.bernoulli(key, fire_rate, shape=shape)
    mask = mask[..., None]  # Broadcast to all channels
    return state + ds * mask


def alive_masking(
    state: jnp.ndarray,
    alpha_channel: int = 3,
    threshold: float = 0.1
) -> jnp.ndarray:
    """Zero out dead cells based on alpha channel neighborhood.

    A cell is alive if any cell in its 3x3 neighborhood has alpha > threshold.

    Args:
        state: Cell state of shape (H, W, C) or (B, H, W, C)
        alpha_channel: Index of alpha channel
        threshold: Alive threshold

    Returns:
        State with dead cells zeroed
    """
    has_batch = state.ndim == 4

    alpha = state[..., alpha_channel:alpha_channel + 1]

    if has_batch:
        window_shape = (1, 3, 3, 1)
        strides = (1, 1, 1, 1)
    else:
        window_shape = (3, 3, 1)
        strides = (1, 1, 1)

    alive = jax.lax.reduce_window(
        alpha,
        -jnp.inf,
        jax.lax.max,
        window_shape,
        strides,
        'SAME'
    ) > threshold

    return state * alive.astype(jnp.float32)


def advect_alpha(
    alpha: jnp.ndarray,
    velocity_x: jnp.ndarray,
    velocity_y: jnp.ndarray,
    dt: float = 0.5
) -> jnp.ndarray:
    """Transport alpha (mass) based on velocity field.

    This is a first-order upwind advection scheme that ensures:
    - Mass conservation: total alpha is preserved
    - Stability: uses upwind differencing
    - Locality: only considers immediate neighbors

    Args:
        alpha: Mass/density field of shape (H, W) or (B, H, W)
        velocity_x: Horizontal velocity, positive = rightward
        velocity_y: Vertical velocity, positive = downward
        dt: Time step (should be <= 0.5 for stability)

    Returns:
        Updated alpha field with mass transported according to velocity
    """
    # Clamp velocities for CFL stability condition
    velocity_x = jnp.clip(velocity_x, -1.0, 1.0)
    velocity_y = jnp.clip(velocity_y, -1.0, 1.0)

    # Compute outflow from each cell based on velocity direction
    # Positive vx = flow right, negative vx = flow left
    outflow_right = alpha * jnp.maximum(velocity_x, 0.0)
    outflow_left = alpha * jnp.maximum(-velocity_x, 0.0)
    outflow_down = alpha * jnp.maximum(velocity_y, 0.0)
    outflow_up = alpha * jnp.maximum(-velocity_y, 0.0)

    # Total mass leaving this cell
    total_outflow = outflow_right + outflow_left + outflow_down + outflow_up

    # Compute inflow from neighbors
    # Roll brings neighbor values to current position
    # Left neighbor's rightward flow arrives here
    inflow_from_left = jnp.roll(outflow_right, 1, axis=-1)
    # Right neighbor's leftward flow arrives here
    inflow_from_right = jnp.roll(outflow_left, -1, axis=-1)
    # Upper neighbor's downward flow arrives here
    inflow_from_up = jnp.roll(outflow_down, 1, axis=-2)
    # Lower neighbor's upward flow arrives here
    inflow_from_down = jnp.roll(outflow_up, -1, axis=-2)

    # Total mass arriving at this cell
    total_inflow = inflow_from_left + inflow_from_right + inflow_from_up + inflow_from_down

    # Update: subtract outflow, add inflow
    new_alpha = alpha + dt * (total_inflow - total_outflow)

    # Clamp to valid range (small epsilon to prevent exactly 0 for gradients)
    return jnp.clip(new_alpha, 0.0, 1.0)


def advect_state(
    state: jnp.ndarray,
    velocity_x_channel: int = 7,
    velocity_y_channel: int = 8,
    alpha_channel: int = 3,
    dt: float = 0.5,
    advect_all: bool = False
) -> jnp.ndarray:
    """Advect alpha (and optionally all channels) based on velocity channels.

    Args:
        state: Full state tensor (H, W, C) or (B, H, W, C)
        velocity_x_channel: Index of x-velocity channel
        velocity_y_channel: Index of y-velocity channel
        alpha_channel: Index of alpha channel
        dt: Time step for advection
        advect_all: If True, advect all channels; if False, only alpha

    Returns:
        State with advected alpha (and optionally other channels)
    """
    vx = state[..., velocity_x_channel]
    vy = state[..., velocity_y_channel]

    if advect_all:
        # Advect all channels (mass-weighted transport)
        new_channels = []
        for c in range(state.shape[-1]):
            channel = state[..., c]
            advected = advect_alpha(channel, vx, vy, dt)
            new_channels.append(advected)
        return jnp.stack(new_channels, axis=-1)
    else:
        # Only advect alpha channel
        alpha = state[..., alpha_channel]
        new_alpha = advect_alpha(alpha, vx, vy, dt)
        return state.at[..., alpha_channel].set(new_alpha)


def soft_clamp(
    x: jnp.ndarray,
    min_val: float = -3.0,
    max_val: float = 3.0
) -> jnp.ndarray:
    """Soft clamping using tanh for gradient-friendly bounds.

    Args:
        x: Input tensor
        min_val: Minimum value
        max_val: Maximum value

    Returns:
        Soft-clamped tensor
    """
    scale = (max_val - min_val) / 2
    offset = (max_val + min_val) / 2
    return scale * jnp.tanh((x - offset) / scale) + offset


class NCA(nn.Module):
    """Base Neural Cellular Automata module.

    Implements the core NCA loop: perceive -> update -> stochastic mask -> alive mask

    Attributes:
        num_channels: Number of state channels
        hidden_dim: Hidden layer dimension in update rule
        fire_rate: Probability of cell update (stochastic mask)
        alpha_channel: Index of alpha (alive) channel
        alive_threshold: Threshold for alive masking
        use_circular_padding: Whether to use circular padding
    """
    num_channels: int = 16
    hidden_dim: int = 128
    fire_rate: float = 0.5
    alpha_channel: int = 3
    alive_threshold: float = 0.1
    use_circular_padding: bool = True

    def setup(self):
        self.perceive = DepthwiseConvPerceive(
            num_channels=self.num_channels,
            use_circular_padding=self.use_circular_padding
        )
        self.update_rule = NCAUpdateRule(
            num_channels=self.num_channels,
            hidden_dim=self.hidden_dim
        )

    def __call__(
        self,
        state: jnp.ndarray,
        key: jax.random.PRNGKey
    ) -> jnp.ndarray:
        """Execute one NCA step.

        Args:
            state: Current state of shape (H, W, C) or (B, H, W, C)
            key: PRNG key for stochastic update

        Returns:
            Updated state
        """
        # Perceive
        perception = self.perceive(state)

        # Compute update
        ds = self.update_rule(perception)

        # Stochastic update
        state = stochastic_update(state, ds, key, self.fire_rate)

        # Alive masking
        state = alive_masking(state, self.alpha_channel, self.alive_threshold)

        return state

    def multi_step(
        self,
        state: jnp.ndarray,
        key: jax.random.PRNGKey,
        num_steps: int
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Run multiple NCA steps using scan for memory efficiency.

        Args:
            state: Initial state
            key: PRNG key
            num_steps: Number of steps to run

        Returns:
            Tuple of (final_state, trajectory)
        """
        keys = jax.random.split(key, num_steps)

        def step_fn(carry, subkey):
            state = self(carry, subkey)
            return state, state

        final_state, trajectory = jax.lax.scan(step_fn, state, keys)
        return final_state, trajectory


def create_seed(
    height: int,
    width: int,
    channels: int = 16,
    center: bool = True
) -> jnp.ndarray:
    """Create a seed state with a single active cell.

    Args:
        height: Grid height
        width: Grid width
        channels: Number of channels
        center: Whether to place seed in center (else random)

    Returns:
        Seed state tensor
    """
    seed = jnp.zeros((height, width, channels))

    if center:
        cy, cx = height // 2, width // 2
    else:
        cy, cx = height // 2, width // 2  # Still center for determinism

    # Set alpha (channel 3) and hidden channels to 1.0
    seed = seed.at[cy, cx, 3:].set(1.0)

    return seed


def create_multi_seed(
    height: int,
    width: int,
    channels: int = 16,
    num_seeds: int = 2,
    separation: int = 20
) -> jnp.ndarray:
    """Create seed state with multiple active cells for symmetry breaking.

    Args:
        height: Grid height
        width: Grid width
        channels: Number of channels
        num_seeds: Number of seed cells
        separation: Distance between seeds

    Returns:
        Seed state tensor
    """
    seed = jnp.zeros((height, width, channels))

    cy = height // 2
    cx_start = width // 2 - (separation * (num_seeds - 1)) // 2

    for i in range(num_seeds):
        cx = cx_start + i * separation
        # Vary hidden channel initialization for each seed
        hidden_val = 1.0 - i * 0.2
        seed = seed.at[cy, cx, 3:].set(hidden_val)

    return seed


def create_genome_seed(
    height: int,
    width: int,
    channels: int = 16,
    genome_bits: int = 4,
    target_id: int = 0
) -> jnp.ndarray:
    """Create seed with genome encoding for multi-target NCA.

    Args:
        height: Grid height
        width: Grid width
        channels: Number of channels
        genome_bits: Number of bits for genome encoding
        target_id: Target formation ID to encode

    Returns:
        Seed state tensor with encoded genome
    """
    seed = jnp.zeros((height, width, channels))

    cy, cx = height // 2, width // 2

    # Encode target ID in first genome_bits hidden channels (after alpha)
    genome = jnp.array([(target_id >> i) & 1 for i in range(genome_bits)],
                       dtype=jnp.float32)
    seed = seed.at[cy, cx, 4:4 + genome_bits].set(genome)
    seed = seed.at[cy, cx, 3].set(1.0)  # Alpha
    seed = seed.at[cy, cx, 4 + genome_bits:].set(1.0)  # Remaining hidden

    return seed
