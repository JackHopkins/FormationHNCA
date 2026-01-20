"""Advection-based Neural Cellular Automata.

This NCA variant uses physical mass transport instead of direct alpha updates:
- The NCA outputs velocity (direction to move)
- Mass (alpha) is transported via advection physics
- Total mass is conserved by construction

This is more physically plausible for simulating soldiers:
- A cell is occupied or not (binary-ish)
- Soldiers move from cell to cell
- No soldiers created/destroyed during movement
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from dataclasses import dataclass

from battle_nca.core.perceive import MultiScalePerceive
from battle_nca.core.nca import stochastic_update
from battle_nca.core.advection import advect_mass, diffuse_mass, add_velocity_noise


@dataclass
class AdvectionChannels:
    """Channel allocation for advection-based NCA.

    Simplified from ChildChannels - focused on mass transport.

    Channels:
        0-2: RGB visualization
        3: Mass/occupancy (transported via advection)
        4-5: Velocity (vx, vy) - controls mass transport
        6-7: Target velocity (from parent signal)
        8-15: Hidden state for coordination
    """
    RGB_START: int = 0
    RGB_END: int = 3
    MASS: int = 3          # Renamed from ALPHA - this is what gets advected
    VELOCITY_X: int = 4
    VELOCITY_Y: int = 5
    TARGET_VX: int = 6     # Target velocity from signal
    TARGET_VY: int = 7
    HIDDEN_START: int = 8
    HIDDEN_END: int = 16

    TOTAL: int = 16


ADVECTION_CHANNELS = AdvectionChannels()


class VelocityUpdateRule(nn.Module):
    """Update rule that outputs velocity for mass transport.

    The key difference from standard NCA:
    - Does NOT output mass/alpha updates directly
    - Outputs velocity that determines how mass moves
    - Mass transport happens via advection physics
    """
    num_channels: int = 16
    hidden_dim: int = 64

    @nn.compact
    def __call__(
        self,
        perception: jnp.ndarray,
        parent_signal: jnp.ndarray | None = None
    ) -> jnp.ndarray:
        """Compute velocity and hidden state updates.

        Args:
            perception: Multi-scale perception tensor
            parent_signal: Optional goal signal (target velocity, etc.)

        Returns:
            Update tensor (same shape as state)
        """
        # Combine perception with parent signal
        if parent_signal is not None:
            x = jnp.concatenate([perception, parent_signal], axis=-1)
        else:
            x = perception

        # Hidden layers
        x = nn.Conv(self.hidden_dim, (1, 1), name='hidden1')(x)
        x = nn.relu(x)
        x = nn.Conv(self.hidden_dim // 2, (1, 1), name='hidden2')(x)
        x = nn.relu(x)

        # RGB update (for visualization)
        rgb_update = nn.Conv(
            3, (1, 1),
            kernel_init=nn.initializers.zeros,
            name='rgb_out'
        )(x)

        # NO mass update - mass is moved by advection only
        mass_update = jnp.zeros(x.shape[:-1] + (1,))

        # Velocity update - THIS is what the NCA learns
        # The network learns what velocity to set to achieve the goal
        velocity_update = nn.Conv(
            2, (1, 1),
            kernel_init=nn.initializers.zeros,
            name='velocity_out'
        )(x)

        # Target velocity channels (pass through from signal)
        target_v_update = jnp.zeros(x.shape[:-1] + (2,))

        # Hidden state update
        hidden_update = nn.Conv(
            ADVECTION_CHANNELS.HIDDEN_END - ADVECTION_CHANNELS.HIDDEN_START,
            (1, 1),
            kernel_init=nn.initializers.zeros,
            name='hidden_out'
        )(x)

        # Assemble full update
        full_update = jnp.concatenate([
            rgb_update,      # 0-2
            mass_update,     # 3 (always zero - advection handles this)
            velocity_update, # 4-5
            target_v_update, # 6-7
            hidden_update    # 8-15
        ], axis=-1)

        return full_update


class AdvectionNCA(nn.Module):
    """NCA with advection-based mass transport.

    Instead of directly updating mass/alpha, this NCA:
    1. Perceives local neighborhood
    2. Computes velocity updates (where should mass go?)
    3. Applies advection to transport mass according to velocity
    4. Applies diffusion for exploration

    This ensures mass conservation and makes movement explicit.

    Attributes:
        num_channels: Number of state channels (default 16)
        hidden_dim: Hidden layer dimension
        fire_rate: Stochastic update probability
        advection_dt: Time step for advection (smaller = more stable)
        advection_steps: Number of advection sub-steps per NCA step
        diffusion_rate: Rate of mass diffusion for exploration (0 = none)
        velocity_noise: Scale of random noise added to velocity (0 = none)
    """
    num_channels: int = 16
    hidden_dim: int = 64
    fire_rate: float = 0.5
    advection_dt: float = 0.25
    advection_steps: int = 2  # Multiple small steps for stability
    diffusion_rate: float = 0.05  # Spread mass for exploration
    velocity_noise: float = 0.2   # Random velocity perturbation
    velocity_damping: float = 0.95  # Velocity decay per step (1.0 = no damping)

    def setup(self):
        self.perceive = MultiScalePerceive(
            num_channels=self.num_channels,
            use_circular_padding=True
        )
        self.update_rule = VelocityUpdateRule(
            num_channels=self.num_channels,
            hidden_dim=self.hidden_dim
        )

    def __call__(
        self,
        state: jnp.ndarray,
        key: jax.random.PRNGKey,
        parent_signal: jnp.ndarray | None = None
    ) -> jnp.ndarray:
        """Execute one advection-NCA step.

        Args:
            state: Current state (H, W, C) or (B, H, W, C)
            key: PRNG key for stochastic update
            parent_signal: Optional goal signal (target formation, etc.)

        Returns:
            Updated state with mass transported via advection
        """
        # 1. Perceive neighborhood
        perceptions = self.perceive(state)
        # Use formation-scale perception as main input
        perception = perceptions['formation']

        # 2. Compute updates (velocity + hidden state, NOT mass)
        ds = self.update_rule(perception, parent_signal)

        # 3. Apply stochastic update to non-mass channels
        # First, zero out the mass channel in the update
        ds = ds.at[..., ADVECTION_CHANNELS.MASS].set(0.0)
        state = stochastic_update(state, ds, key, self.fire_rate)

        # 4. Clamp velocity to valid range
        state = state.at[..., ADVECTION_CHANNELS.VELOCITY_X].set(
            jnp.clip(state[..., ADVECTION_CHANNELS.VELOCITY_X], -1.0, 1.0)
        )
        state = state.at[..., ADVECTION_CHANNELS.VELOCITY_Y].set(
            jnp.clip(state[..., ADVECTION_CHANNELS.VELOCITY_Y], -1.0, 1.0)
        )

        # 5. Advect mass according to velocity
        mass = state[..., ADVECTION_CHANNELS.MASS]
        vx = state[..., ADVECTION_CHANNELS.VELOCITY_X]
        vy = state[..., ADVECTION_CHANNELS.VELOCITY_Y]

        # Apply velocity damping (helps cells settle at target)
        if self.velocity_damping < 1.0:
            vx = self.velocity_damping * vx
            vy = self.velocity_damping * vy
            # Update state with damped velocity
            state = state.at[..., ADVECTION_CHANNELS.VELOCITY_X].set(vx)
            state = state.at[..., ADVECTION_CHANNELS.VELOCITY_Y].set(vy)

        # Add velocity noise for exploration
        if self.velocity_noise > 0:
            key, noise_key = jax.random.split(key)
            vx, vy = add_velocity_noise(vx, vy, noise_key, self.velocity_noise)

        # Multiple small advection steps for stability
        for _ in range(self.advection_steps):
            mass = advect_mass(mass, vx, vy, self.advection_dt)

        # Apply diffusion for exploration (mass spreads to neighbors)
        if self.diffusion_rate > 0:
            mass = diffuse_mass(mass, self.diffusion_rate)

        state = state.at[..., ADVECTION_CHANNELS.MASS].set(mass)

        # 6. Clamp RGB and hidden channels
        state = state.at[..., :3].set(jnp.clip(state[..., :3], 0.0, 1.0))

        return state


def create_advection_seed(
    height: int,
    width: int,
    spawn_region: tuple[int, int, int, int] | None = None,
    mass_value: float = 1.0
) -> jnp.ndarray:
    """Create initial state for advection NCA.

    Args:
        height: Grid height
        width: Grid width
        spawn_region: (y_start, y_end, x_start, x_end) or None for center
        mass_value: Initial mass value in spawn region

    Returns:
        Initial state tensor
    """
    state = jnp.zeros((height, width, ADVECTION_CHANNELS.TOTAL))

    if spawn_region is None:
        # Default: small center region
        y_start = height // 2 - 2
        y_end = height // 2 + 2
        x_start = width // 2 - 2
        x_end = width // 2 + 2
    else:
        y_start, y_end, x_start, x_end = spawn_region

    # Set mass in spawn region
    state = state.at[y_start:y_end, x_start:x_end, ADVECTION_CHANNELS.MASS].set(mass_value)

    # Set RGB to red for visualization
    state = state.at[y_start:y_end, x_start:x_end, 0].set(1.0)  # R

    # Initialize hidden channels with small values
    state = state.at[y_start:y_end, x_start:x_end, ADVECTION_CHANNELS.HIDDEN_START:].set(0.1)

    return state


def create_formation_from_alpha(
    alpha: jnp.ndarray,
    rgb: tuple[float, float, float] = (1.0, 0.0, 0.0)
) -> jnp.ndarray:
    """Create full state from alpha/mass pattern.

    Args:
        alpha: 2D mass pattern (H, W)
        rgb: RGB color for visualization

    Returns:
        Full state tensor (H, W, C)
    """
    h, w = alpha.shape
    state = jnp.zeros((h, w, ADVECTION_CHANNELS.TOTAL))

    # Set RGB where mass exists
    state = state.at[..., 0].set(rgb[0] * (alpha > 0.1))
    state = state.at[..., 1].set(rgb[1] * (alpha > 0.1))
    state = state.at[..., 2].set(rgb[2] * (alpha > 0.1))

    # Set mass
    state = state.at[..., ADVECTION_CHANNELS.MASS].set(alpha)

    # Initialize hidden channels
    state = state.at[..., ADVECTION_CHANNELS.HIDDEN_START:].set(
        0.1 * (alpha > 0.1)[..., None]
    )

    return state