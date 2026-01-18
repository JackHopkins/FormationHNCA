"""Child-NCA for per-unit battle simulation."""

import jax
import jax.numpy as jnp
from flax import linen as nn
from dataclasses import dataclass

from battle_nca.core.perceive import perceive, MultiScalePerceive
from battle_nca.core.nca import stochastic_update, alive_masking


@dataclass
class ChildChannels:
    """Channel allocation for child-NCA (24 channels total).

    Channels:
        0-2: RGB visualization (team colors)
        3: Alpha/alive (unit presence)
        4: Health [0, 1]
        5: Morale [-1, 1], negative = routing
        6: Fatigue [0, 1], 0 = fresh
        7-8: Velocity (vx, vy) normalized
        9: Unit type encoding
        10: Formation ID
        11-12: Parent command signals
        13-14: Enemy proximity/direction
        15-23: Hidden state channels
    """
    RGB_START: int = 0
    RGB_END: int = 3
    ALPHA: int = 3
    HEALTH: int = 4
    MORALE: int = 5
    FATIGUE: int = 6
    VELOCITY_X: int = 7
    VELOCITY_Y: int = 8
    UNIT_TYPE: int = 9
    FORMATION_ID: int = 10
    PARENT_SIGNAL_START: int = 11
    PARENT_SIGNAL_END: int = 13
    ENEMY_PROXIMITY: int = 13
    ENEMY_DIRECTION: int = 14
    HIDDEN_START: int = 15
    HIDDEN_END: int = 24

    TOTAL: int = 24


CHILD_CHANNELS = ChildChannels()


class ChildUpdateRule(nn.Module):
    """Update rule for child-NCA with battle-specific pathways.

    Processes multi-scale perception and parent signals to produce
    state updates for combat, movement, and coordination.

    Attributes:
        num_channels: Total number of output channels
        hidden_dim: Hidden layer dimension
    """
    num_channels: int = 24
    hidden_dim: int = 128

    @nn.compact
    def __call__(
        self,
        melee_perception: jnp.ndarray,
        morale_perception: jnp.ndarray,
        formation_perception: jnp.ndarray,
        parent_signal: jnp.ndarray | None = None
    ) -> jnp.ndarray:
        """Compute state update from multi-scale perception.

        Args:
            melee_perception: 3x3 Sobel perception for combat
            morale_perception: 7x7 smoothed perception for morale contagion
            formation_perception: 11x11 smoothed for formation cohesion
            parent_signal: Optional parent NCA commands

        Returns:
            Residual state update
        """
        # Combat pathway
        combat_in = melee_perception
        if parent_signal is not None:
            combat_in = jnp.concatenate([combat_in, parent_signal], axis=-1)

        combat_hidden = nn.Conv(64, (1, 1), name='combat_h1')(combat_in)
        combat_hidden = nn.relu(combat_hidden)
        combat_hidden = nn.Conv(32, (1, 1), name='combat_h2')(combat_hidden)
        combat_hidden = nn.relu(combat_hidden)

        # Health update
        health_update = nn.Conv(
            1, (1, 1),
            kernel_init=nn.initializers.zeros,
            name='health_out'
        )(combat_hidden)

        # Morale pathway
        morale_in = jnp.concatenate([
            melee_perception,
            morale_perception
        ], axis=-1)

        morale_hidden = nn.Conv(64, (1, 1), name='morale_h1')(morale_in)
        morale_hidden = nn.relu(morale_hidden)

        morale_update = nn.Conv(
            1, (1, 1),
            kernel_init=nn.initializers.zeros,
            name='morale_out'
        )(morale_hidden)

        # Fatigue update (simple decay based on activity)
        fatigue_update = nn.Conv(
            1, (1, 1),
            kernel_init=nn.initializers.zeros,
            name='fatigue_out'
        )(combat_hidden)

        # Movement pathway
        movement_in = jnp.concatenate([
            melee_perception,
            formation_perception
        ], axis=-1)

        if parent_signal is not None:
            movement_in = jnp.concatenate([movement_in, parent_signal], axis=-1)

        movement_hidden = nn.Conv(64, (1, 1), name='movement_h1')(movement_in)
        movement_hidden = nn.relu(movement_hidden)

        velocity_update = nn.Conv(
            2, (1, 1),
            kernel_init=nn.initializers.zeros,
            name='velocity_out'
        )(movement_hidden)

        # Visualization channels (RGB)
        rgb_update = nn.Conv(
            3, (1, 1),
            kernel_init=nn.initializers.zeros,
            name='rgb_out'
        )(combat_hidden)

        # Alpha update
        alpha_update = nn.Conv(
            1, (1, 1),
            kernel_init=nn.initializers.zeros,
            name='alpha_out'
        )(combat_hidden)

        # Static channels (type, formation ID) - no update
        static_update = jnp.zeros(
            (*melee_perception.shape[:-1], 2),
            dtype=melee_perception.dtype
        )

        # Parent signal channels - written by actuator, zero update here
        parent_signal_update = jnp.zeros(
            (*melee_perception.shape[:-1], 2),
            dtype=melee_perception.dtype
        )

        # Enemy info channels - computed externally
        enemy_update = jnp.zeros(
            (*melee_perception.shape[:-1], 2),
            dtype=melee_perception.dtype
        )

        # Hidden state pathway
        hidden_in = jnp.concatenate([
            melee_perception,
            morale_perception,
            formation_perception
        ], axis=-1)

        hidden_h = nn.Conv(self.hidden_dim, (1, 1), name='hidden_h1')(hidden_in)
        hidden_h = nn.relu(hidden_h)

        hidden_update = nn.Conv(
            CHILD_CHANNELS.HIDDEN_END - CHILD_CHANNELS.HIDDEN_START,
            (1, 1),
            kernel_init=nn.initializers.zeros,
            name='hidden_out'
        )(hidden_h)

        # Combine all updates in channel order
        full_update = jnp.concatenate([
            rgb_update,            # 0-2
            alpha_update,          # 3
            health_update,         # 4
            morale_update,         # 5
            fatigue_update,        # 6
            velocity_update,       # 7-8
            static_update,         # 9-10
            parent_signal_update,  # 11-12
            enemy_update,          # 13-14
            hidden_update          # 15-23
        ], axis=-1)

        return full_update


class ChildNCA(nn.Module):
    """Child-NCA for per-unit level battle simulation.

    Each cell represents a single combat unit with 24-channel state
    encoding combat stats, movement, and coordination signals.

    Attributes:
        num_channels: Number of state channels (default 24)
        hidden_dim: Hidden layer dimension
        fire_rate: Stochastic update probability
        use_circular_padding: Whether to use circular padding
    """
    num_channels: int = 24
    hidden_dim: int = 128
    fire_rate: float = 0.5
    use_circular_padding: bool = True

    def setup(self):
        self.multi_perceive = MultiScalePerceive(
            num_channels=self.num_channels,
            use_circular_padding=self.use_circular_padding
        )
        self.update_rule = ChildUpdateRule(
            num_channels=self.num_channels,
            hidden_dim=self.hidden_dim
        )

    def __call__(
        self,
        state: jnp.ndarray,
        key: jax.random.PRNGKey,
        parent_signal: jnp.ndarray | None = None,
        enemy_state: jnp.ndarray | None = None
    ) -> jnp.ndarray:
        """Execute one child-NCA step.

        Args:
            state: Current state (H, W, 24) or (B, H, W, 24)
            key: PRNG key
            parent_signal: Parent NCA commands (upsampled to child resolution)
            enemy_state: Enemy army state for combat computation

        Returns:
            Updated state
        """
        # Compute multi-scale perception
        perceptions = self.multi_perceive(state)

        # Update enemy proximity if enemy state provided
        if enemy_state is not None:
            state = self._update_enemy_info(state, enemy_state)

        # Compute update
        ds = self.update_rule(
            perceptions['melee'],
            perceptions['morale'],
            perceptions['formation'],
            parent_signal
        )

        # Stochastic update
        state = stochastic_update(state, ds, key, self.fire_rate)

        # Alive masking
        state = alive_masking(state, CHILD_CHANNELS.ALPHA, threshold=0.1)

        # Clamp specific channels to valid ranges
        state = self._clamp_channels(state)

        return state

    def _update_enemy_info(
        self,
        state: jnp.ndarray,
        enemy_state: jnp.ndarray
    ) -> jnp.ndarray:
        """Update enemy proximity and direction channels.

        Args:
            state: Current army state
            enemy_state: Enemy army state

        Returns:
            State with updated enemy info channels
        """
        # Enemy presence via max pooling of alpha
        enemy_alpha = enemy_state[..., CHILD_CHANNELS.ALPHA:CHILD_CHANNELS.ALPHA + 1]

        has_batch = enemy_alpha.ndim == 4
        if has_batch:
            window = (1, 5, 5, 1)
            strides = (1, 1, 1, 1)
        else:
            window = (5, 5, 1)
            strides = (1, 1, 1)

        enemy_proximity = jax.lax.reduce_window(
            enemy_alpha, -jnp.inf, jax.lax.max, window, strides, 'SAME'
        )

        # Simple direction estimation (center of mass of nearby enemies)
        # This is a simplified version - full version would use proper gradient
        enemy_direction = jnp.zeros_like(enemy_proximity)

        state = state.at[..., CHILD_CHANNELS.ENEMY_PROXIMITY].set(
            enemy_proximity[..., 0]
        )
        state = state.at[..., CHILD_CHANNELS.ENEMY_DIRECTION].set(
            enemy_direction[..., 0]
        )

        return state

    def _clamp_channels(self, state: jnp.ndarray) -> jnp.ndarray:
        """Clamp channel values to valid ranges.

        Args:
            state: Current state

        Returns:
            State with clamped channels
        """
        # RGB and alpha: [0, 1]
        state = state.at[..., :4].set(
            jnp.clip(state[..., :4], 0.0, 1.0)
        )
        # Health: [0, 1]
        state = state.at[..., CHILD_CHANNELS.HEALTH].set(
            jnp.clip(state[..., CHILD_CHANNELS.HEALTH], 0.0, 1.0)
        )
        # Morale: [-1, 1]
        state = state.at[..., CHILD_CHANNELS.MORALE].set(
            jnp.clip(state[..., CHILD_CHANNELS.MORALE], -1.0, 1.0)
        )
        # Fatigue: [0, 1]
        state = state.at[..., CHILD_CHANNELS.FATIGUE].set(
            jnp.clip(state[..., CHILD_CHANNELS.FATIGUE], 0.0, 1.0)
        )
        # Velocity: [-1, 1]
        state = state.at[..., CHILD_CHANNELS.VELOCITY_X:CHILD_CHANNELS.VELOCITY_Y+1].set(
            jnp.clip(state[..., CHILD_CHANNELS.VELOCITY_X:CHILD_CHANNELS.VELOCITY_Y+1], -1.0, 1.0)
        )
        # Hidden channels: [-2, 2] (allow some range but prevent explosion)
        state = state.at[..., CHILD_CHANNELS.HIDDEN_START:CHILD_CHANNELS.HIDDEN_END].set(
            jnp.clip(state[..., CHILD_CHANNELS.HIDDEN_START:CHILD_CHANNELS.HIDDEN_END], -2.0, 2.0)
        )

        return state

    def multi_step(
        self,
        state: jnp.ndarray,
        key: jax.random.PRNGKey,
        num_steps: int,
        parent_signal: jnp.ndarray | None = None,
        enemy_state: jnp.ndarray | None = None
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Run multiple child-NCA steps.

        Args:
            state: Initial state
            key: PRNG key
            num_steps: Number of steps
            parent_signal: Parent commands (constant for all steps)
            enemy_state: Enemy state (constant for all steps)

        Returns:
            Tuple of (final_state, trajectory)
        """
        keys = jax.random.split(key, num_steps)

        def step_fn(carry, subkey):
            new_state = self(carry, subkey, parent_signal, enemy_state)
            return new_state, new_state

        final_state, trajectory = jax.lax.scan(step_fn, state, keys)
        return final_state, trajectory


def create_army_seed(
    height: int,
    width: int,
    team_color: tuple[float, float, float] = (1.0, 0.0, 0.0),
    unit_type: int = 0,
    formation_id: int = 0,
    spawn_region: tuple[int, int, int, int] | None = None
) -> jnp.ndarray:
    """Create initial seed state for an army.

    Args:
        height: Grid height
        width: Grid width
        team_color: RGB team color
        unit_type: Unit type encoding
        formation_id: Formation ID
        spawn_region: (y_start, y_end, x_start, x_end) or None for center

    Returns:
        Army seed state
    """
    state = jnp.zeros((height, width, CHILD_CHANNELS.TOTAL))

    if spawn_region is None:
        # Center spawn
        cy, cx = height // 2, width // 2
        spawn_region = (cy - 2, cy + 2, cx - 2, cx + 2)

    y_start, y_end, x_start, x_end = spawn_region

    # Set team color
    state = state.at[y_start:y_end, x_start:x_end, 0].set(team_color[0])
    state = state.at[y_start:y_end, x_start:x_end, 1].set(team_color[1])
    state = state.at[y_start:y_end, x_start:x_end, 2].set(team_color[2])

    # Alpha = 1 (alive)
    state = state.at[y_start:y_end, x_start:x_end, CHILD_CHANNELS.ALPHA].set(1.0)

    # Health = 1 (full)
    state = state.at[y_start:y_end, x_start:x_end, CHILD_CHANNELS.HEALTH].set(1.0)

    # Morale = 0.5 (neutral)
    state = state.at[y_start:y_end, x_start:x_end, CHILD_CHANNELS.MORALE].set(0.5)

    # Fatigue = 0 (fresh)
    state = state.at[y_start:y_end, x_start:x_end, CHILD_CHANNELS.FATIGUE].set(0.0)

    # Unit type and formation
    state = state.at[y_start:y_end, x_start:x_end, CHILD_CHANNELS.UNIT_TYPE].set(
        float(unit_type)
    )
    state = state.at[y_start:y_end, x_start:x_end, CHILD_CHANNELS.FORMATION_ID].set(
        float(formation_id)
    )

    # Initialize hidden channels with small random-like values
    hidden_init = jnp.ones((y_end - y_start, x_end - x_start,
                            CHILD_CHANNELS.HIDDEN_END - CHILD_CHANNELS.HIDDEN_START)) * 0.1
    state = state.at[
        y_start:y_end,
        x_start:x_end,
        CHILD_CHANNELS.HIDDEN_START:CHILD_CHANNELS.HIDDEN_END
    ].set(hidden_init)

    return state
