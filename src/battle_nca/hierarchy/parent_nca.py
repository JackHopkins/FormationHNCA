"""Parent-NCA for formation-level control."""

import jax
import jax.numpy as jnp
from flax import linen as nn
from dataclasses import dataclass

from battle_nca.core.perceive import perceive
from battle_nca.core.nca import stochastic_update, alive_masking


@dataclass
class ParentChannels:
    """Channel allocation for parent-NCA (16 channels total).

    Channels:
        0-3: Formation shape (RGBA target)
        4-5: Formation velocity/heading
        6: Formation integrity (% alive units)
        7-8: Command outputs (advance/hold/charge/wheel encoded)
        9-15: Hidden coordination state
    """
    RGBA_START: int = 0
    RGBA_END: int = 4
    VELOCITY_X: int = 4
    VELOCITY_Y: int = 5
    INTEGRITY: int = 6
    COMMAND_START: int = 7
    COMMAND_END: int = 9
    HIDDEN_START: int = 9
    HIDDEN_END: int = 16

    TOTAL: int = 16


PARENT_CHANNELS = ParentChannels()


# Command encodings
class Commands:
    """Command signal encodings for parent-to-child communication."""
    HOLD = 0
    ADVANCE = 1
    CHARGE = 2
    RETREAT = 3
    WHEEL_LEFT = 4
    WHEEL_RIGHT = 5

    @staticmethod
    def encode(command: int) -> tuple[float, float]:
        """Encode command as two-channel signal.

        Args:
            command: Command ID

        Returns:
            Tuple of (signal_1, signal_2)
        """
        encodings = {
            Commands.HOLD: (0.0, 0.0),
            Commands.ADVANCE: (1.0, 0.0),
            Commands.CHARGE: (1.0, 1.0),
            Commands.RETREAT: (-1.0, 0.0),
            Commands.WHEEL_LEFT: (0.0, 1.0),
            Commands.WHEEL_RIGHT: (0.0, -1.0),
        }
        return encodings.get(command, (0.0, 0.0))


class ParentUpdateRule(nn.Module):
    """Update rule for parent-NCA controlling formations.

    Attributes:
        num_channels: Number of output channels
        hidden_dim: Hidden layer dimension
    """
    num_channels: int = 16
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, perception: jnp.ndarray) -> jnp.ndarray:
        """Compute formation-level state update.

        Args:
            perception: Perception tensor from parent grid

        Returns:
            Residual state update
        """
        x = perception

        # Shared hidden
        x = nn.Conv(self.hidden_dim, (1, 1), name='hidden1')(x)
        x = nn.relu(x)
        x = nn.Conv(self.hidden_dim // 2, (1, 1), name='hidden2')(x)
        x = nn.relu(x)

        # Formation shape update (RGBA)
        shape_update = nn.Conv(
            4, (1, 1),
            kernel_init=nn.initializers.zeros,
            name='shape_out'
        )(x)

        # Velocity update
        velocity_update = nn.Conv(
            2, (1, 1),
            kernel_init=nn.initializers.zeros,
            name='velocity_out'
        )(x)

        # Integrity (computed from sensor, minimal update)
        integrity_update = nn.Conv(
            1, (1, 1),
            kernel_init=nn.initializers.zeros,
            name='integrity_out'
        )(x)

        # Command outputs
        command_update = nn.Conv(
            2, (1, 1),
            kernel_init=nn.initializers.zeros,
            name='command_out'
        )(x)

        # Hidden state
        hidden_update = nn.Conv(
            PARENT_CHANNELS.HIDDEN_END - PARENT_CHANNELS.HIDDEN_START,
            (1, 1),
            kernel_init=nn.initializers.zeros,
            name='hidden_out'
        )(x)

        # Combine in channel order
        full_update = jnp.concatenate([
            shape_update,      # 0-3
            velocity_update,   # 4-5
            integrity_update,  # 6
            command_update,    # 7-8
            hidden_update      # 9-15
        ], axis=-1)

        return full_update


class ParentNCA(nn.Module):
    """Parent-NCA for formation-level coordination.

    Operates on a coarser grid than child-NCA, where each parent cell
    corresponds to a cluster of child cells. Outputs command signals
    that influence child behavior.

    Attributes:
        num_channels: Number of state channels
        hidden_dim: Hidden layer dimension
        fire_rate: Stochastic update probability
        use_circular_padding: Whether to use circular padding
    """
    num_channels: int = 16
    hidden_dim: int = 64
    fire_rate: float = 0.5
    use_circular_padding: bool = True

    def setup(self):
        self.update_rule = ParentUpdateRule(
            num_channels=self.num_channels,
            hidden_dim=self.hidden_dim
        )

    def __call__(
        self,
        state: jnp.ndarray,
        key: jax.random.PRNGKey,
        sensor_input: jnp.ndarray | None = None
    ) -> jnp.ndarray:
        """Execute one parent-NCA step.

        Args:
            state: Current parent state (H_p, W_p, 16) or (B, H_p, W_p, 16)
            key: PRNG key
            sensor_input: Aggregated child state (optional, for initialization)

        Returns:
            Updated parent state
        """
        # If sensor input provided, use it to update relevant channels
        if sensor_input is not None:
            state = self._integrate_sensor(state, sensor_input)

        # Perception
        perception = perceive(state, self.use_circular_padding)

        # Compute update
        ds = self.update_rule(perception)

        # Stochastic update
        state = stochastic_update(state, ds, key, self.fire_rate)

        # Alive masking
        state = alive_masking(state, alpha_channel=3, threshold=0.1)

        return state

    def _integrate_sensor(
        self,
        state: jnp.ndarray,
        sensor_input: jnp.ndarray
    ) -> jnp.ndarray:
        """Integrate sensor information from child cells.

        Args:
            state: Current parent state
            sensor_input: Pooled child state

        Returns:
            State with updated sensor-derived channels
        """
        # Update integrity channel based on child alpha average
        child_alpha_avg = sensor_input[..., 3]  # Child alpha channel
        state = state.at[..., PARENT_CHANNELS.INTEGRITY].set(child_alpha_avg)

        return state

    def get_command_signals(self, state: jnp.ndarray) -> jnp.ndarray:
        """Extract command signals for child actuator.

        Args:
            state: Parent state

        Returns:
            Command signal tensor (H_p, W_p, 2) or (B, H_p, W_p, 2)
        """
        return state[..., PARENT_CHANNELS.COMMAND_START:PARENT_CHANNELS.COMMAND_END]

    def multi_step(
        self,
        state: jnp.ndarray,
        key: jax.random.PRNGKey,
        num_steps: int,
        sensor_input: jnp.ndarray | None = None
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Run multiple parent-NCA steps.

        Args:
            state: Initial parent state
            key: PRNG key
            num_steps: Number of steps
            sensor_input: Optional sensor input (used only on first step)

        Returns:
            Tuple of (final_state, trajectory)
        """
        keys = jax.random.split(key, num_steps)

        def step_fn(carry, step_data):
            idx, subkey = step_data
            # Only use sensor on first step
            sensor = jax.lax.cond(
                idx == 0,
                lambda: sensor_input,
                lambda: None
            ) if sensor_input is not None else None

            new_state = self(carry, subkey, sensor)
            return new_state, new_state

        step_data = (jnp.arange(num_steps), keys)
        final_state, trajectory = jax.lax.scan(
            step_fn, state, step_data
        )
        return final_state, trajectory


def create_parent_seed(
    height: int,
    width: int,
    initial_formation: jnp.ndarray | None = None
) -> jnp.ndarray:
    """Create initial parent-NCA seed state.

    Args:
        height: Parent grid height
        width: Parent grid width
        initial_formation: Optional RGBA formation target

    Returns:
        Parent seed state
    """
    state = jnp.zeros((height, width, PARENT_CHANNELS.TOTAL))

    if initial_formation is not None:
        # Set RGBA from formation
        state = state.at[..., :4].set(initial_formation)
    else:
        # Default: uniform low-alpha
        state = state.at[..., 3].set(0.5)

    # Initialize hidden channels
    state = state.at[..., PARENT_CHANNELS.HIDDEN_START:].set(0.1)

    return state


def create_formation_target(
    height: int,
    width: int,
    formation_type: str = 'line',
    density: float = 0.9
) -> jnp.ndarray:
    """Create formation target pattern for parent-NCA.

    Args:
        height: Parent grid height
        width: Parent grid width
        formation_type: One of 'line', 'phalanx', 'square', 'wedge', 'column'
        density: Target density of units

    Returns:
        RGBA formation target
    """
    target = jnp.zeros((height, width, 4))

    if formation_type == 'line':
        # Thin horizontal line
        row = height // 2
        target = target.at[row, :, 3].set(density)
        target = target.at[row, :, :3].set(1.0)

    elif formation_type == 'phalanx':
        # Deep formation (16 ranks)
        depth = min(16, height // 2)
        start_row = height // 2 - depth // 2
        target = target.at[start_row:start_row + depth, :, 3].set(density)
        target = target.at[start_row:start_row + depth, :, :3].set(1.0)

    elif formation_type == 'square':
        # Hollow square
        thickness = max(2, height // 6)
        # Top and bottom
        target = target.at[:thickness, :, 3].set(density)
        target = target.at[-thickness:, :, 3].set(density)
        # Left and right
        target = target.at[:, :thickness, 3].set(density)
        target = target.at[:, -thickness:, 3].set(density)
        # Set RGB
        target = target.at[..., :3].set(
            jnp.where(target[..., 3:4] > 0, 1.0, 0.0)
        )

    elif formation_type == 'wedge':
        # Triangle pointing forward (up)
        for row in range(height):
            half_width = (height - row) * width // (2 * height)
            center = width // 2
            if half_width > 0:
                target = target.at[row, center - half_width:center + half_width, 3].set(
                    density * (row / height + 0.5)
                )
        target = target.at[..., :3].set(
            jnp.where(target[..., 3:4] > 0, 1.0, 0.0)
        )

    elif formation_type == 'column':
        # Narrow deep column
        col_width = max(2, width // 6)
        center = width // 2
        target = target.at[:, center - col_width:center + col_width, 3].set(density)
        target = target.at[:, center - col_width:center + col_width, :3].set(1.0)

    return target
