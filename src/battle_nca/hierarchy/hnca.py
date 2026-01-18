"""Hierarchical NCA combining parent and child NCAs with sensor/actuator communication."""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import NamedTuple

from battle_nca.hierarchy.child_nca import ChildNCA, CHILD_CHANNELS
from battle_nca.hierarchy.parent_nca import ParentNCA, PARENT_CHANNELS


class HNCAState(NamedTuple):
    """Combined state for hierarchical NCA."""
    child_state: jnp.ndarray  # (H_c, W_c, 24) or (B, H_c, W_c, 24)
    parent_state: jnp.ndarray  # (H_p, W_p, 16) or (B, H_p, W_p, 16)


def sensor(
    child_state: jnp.ndarray,
    cluster_size: int = 4
) -> jnp.ndarray:
    """Aggregate child cell states to parent resolution.

    Implements the sensor component of H-NCA: averages child cell
    clusters to initialize/update parent-NCA state.

    Args:
        child_state: Child state tensor (H_c, W_c, C) or (B, H_c, W_c, C)
        cluster_size: Number of child cells per parent cell (per dimension)

    Returns:
        Aggregated state at parent resolution
    """
    has_batch = child_state.ndim == 4

    if has_batch:
        window = (1, cluster_size, cluster_size, 1)
        strides = (1, cluster_size, cluster_size, 1)
    else:
        window = (cluster_size, cluster_size, 1)
        strides = (cluster_size, cluster_size, 1)

    # Average pooling
    pooled = jax.lax.reduce_window(
        child_state,
        0.0,
        jax.lax.add,
        window,
        strides,
        'VALID'
    ) / (cluster_size ** 2)

    return pooled


def actuator(
    parent_state: jnp.ndarray,
    child_state: jnp.ndarray,
    cluster_size: int = 4,
    signal_channels: tuple[int, int] = (
        CHILD_CHANNELS.PARENT_SIGNAL_START,
        CHILD_CHANNELS.PARENT_SIGNAL_END
    )
) -> jnp.ndarray:
    """Broadcast parent signals to child cells.

    Implements the actuator component of H-NCA: upsamples parent
    state and adds to child signal channels.

    Args:
        parent_state: Parent state tensor
        child_state: Child state tensor
        cluster_size: Upsampling factor
        signal_channels: (start, end) indices for child signal channels

    Returns:
        Child state with injected parent signals
    """
    # Extract command signals from parent
    commands = parent_state[..., PARENT_CHANNELS.COMMAND_START:PARENT_CHANNELS.COMMAND_END]

    # Upsample to child resolution using nearest neighbor
    has_batch = parent_state.ndim == 4

    if has_batch:
        target_shape = (
            commands.shape[0],
            commands.shape[1] * cluster_size,
            commands.shape[2] * cluster_size,
            commands.shape[3]
        )
    else:
        target_shape = (
            commands.shape[0] * cluster_size,
            commands.shape[1] * cluster_size,
            commands.shape[2]
        )

    upsampled = jax.image.resize(
        commands,
        target_shape,
        method='nearest'
    )

    # Ensure shapes match (handle edge cases from pooling)
    if has_batch:
        target_h, target_w = child_state.shape[1:3]
        upsampled = upsampled[:, :target_h, :target_w, :]
    else:
        target_h, target_w = child_state.shape[:2]
        upsampled = upsampled[:target_h, :target_w, :]

    # Add to child signal channels
    start, end = signal_channels
    child_state = child_state.at[..., start:end].add(upsampled)

    return child_state


class HierarchicalNCA(nn.Module):
    """Two-scale Hierarchical NCA for battle simulation.

    Combines parent-NCA (formation control) with child-NCA (unit behavior)
    through sensor/actuator communication.

    Architecture:
        - Child-NCA: 24 channels, per-unit level
        - Parent-NCA: 16 channels, formation level
        - Sensor: Average pooling child → parent
        - Actuator: Nearest neighbor upsample parent → child

    Attributes:
        child_channels: Number of child state channels
        parent_channels: Number of parent state channels
        cluster_size: Child cells per parent cell (per dimension)
        tau_c: Child steps before parent sensing (initial coupling delay)
        child_hidden_dim: Hidden dimension for child update rule
        parent_hidden_dim: Hidden dimension for parent update rule
        fire_rate: Stochastic update probability
    """
    child_channels: int = 24
    parent_channels: int = 16
    cluster_size: int = 4
    tau_c: int = 10
    child_hidden_dim: int = 128
    parent_hidden_dim: int = 64
    fire_rate: float = 0.5

    def setup(self):
        self.child_nca = ChildNCA(
            num_channels=self.child_channels,
            hidden_dim=self.child_hidden_dim,
            fire_rate=self.fire_rate
        )
        self.parent_nca = ParentNCA(
            num_channels=self.parent_channels,
            hidden_dim=self.parent_hidden_dim,
            fire_rate=self.fire_rate
        )

    def __call__(
        self,
        child_state: jnp.ndarray,
        parent_state: jnp.ndarray,
        key: jax.random.PRNGKey,
        enemy_state: jnp.ndarray | None = None
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Execute one coupled H-NCA step.

        Args:
            child_state: Child state tensor
            parent_state: Parent state tensor
            key: PRNG key
            enemy_state: Optional enemy army state

        Returns:
            Tuple of (new_child_state, new_parent_state)
        """
        key1, key2 = jax.random.split(key)

        # Sensor: child → parent
        sensor_input = sensor(child_state, self.cluster_size)

        # Parent step with sensor input
        new_parent_state = self.parent_nca(parent_state, key1, sensor_input)

        # Actuator: parent → child
        child_with_signals = actuator(
            new_parent_state,
            child_state,
            self.cluster_size
        )

        # Child step with parent signals
        parent_signals = child_with_signals[
            ...,
            CHILD_CHANNELS.PARENT_SIGNAL_START:CHILD_CHANNELS.PARENT_SIGNAL_END
        ]
        new_child_state = self.child_nca(
            child_with_signals,
            key2,
            parent_signal=parent_signals,
            enemy_state=enemy_state
        )

        return new_child_state, new_parent_state

    def initial_phase(
        self,
        child_state: jnp.ndarray,
        key: jax.random.PRNGKey,
        enemy_state: jnp.ndarray | None = None
    ) -> jnp.ndarray:
        """Run initial child-only phase (τ_c steps) before coupling.

        Args:
            child_state: Initial child state
            key: PRNG key
            enemy_state: Optional enemy state

        Returns:
            Child state after τ_c steps
        """
        keys = jax.random.split(key, self.tau_c)

        def step_fn(state, subkey):
            new_state = self.child_nca(state, subkey, enemy_state=enemy_state)
            return new_state, None

        final_state, _ = jax.lax.scan(step_fn, child_state, keys)
        return final_state

    def multi_step(
        self,
        child_state: jnp.ndarray,
        parent_state: jnp.ndarray,
        key: jax.random.PRNGKey,
        num_steps: int,
        enemy_state: jnp.ndarray | None = None,
        include_initial_phase: bool = True
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Run multiple H-NCA steps with optional initial phase.

        Args:
            child_state: Initial child state
            parent_state: Initial parent state
            key: PRNG key
            num_steps: Number of coupled steps
            enemy_state: Optional enemy state
            include_initial_phase: Whether to run τ_c child-only steps first

        Returns:
            Tuple of (final_child, final_parent, child_trajectory, parent_trajectory)
        """
        key1, key2 = jax.random.split(key)

        # Initial phase if requested
        if include_initial_phase:
            child_state = self.initial_phase(child_state, key1, enemy_state)

        # Coupled phase
        keys = jax.random.split(key2, num_steps)

        def step_fn(carry, subkey):
            c_state, p_state = carry
            new_c, new_p = self(c_state, p_state, subkey, enemy_state)
            return (new_c, new_p), (new_c, new_p)

        (final_child, final_parent), (child_traj, parent_traj) = jax.lax.scan(
            step_fn,
            (child_state, parent_state),
            keys
        )

        return final_child, final_parent, child_traj, parent_traj


class BattleSimulator(nn.Module):
    """Full battle simulator with two opposing armies.

    Manages two H-NCA systems (red and blue armies) with combat
    interaction between their child states.

    Attributes:
        child_channels: Child state channels
        parent_channels: Parent state channels
        cluster_size: Cells per parent cluster
        tau_c: Initial coupling delay
    """
    child_channels: int = 24
    parent_channels: int = 16
    cluster_size: int = 4
    tau_c: int = 10

    def setup(self):
        self.red_hnca = HierarchicalNCA(
            child_channels=self.child_channels,
            parent_channels=self.parent_channels,
            cluster_size=self.cluster_size,
            tau_c=self.tau_c
        )
        self.blue_hnca = HierarchicalNCA(
            child_channels=self.child_channels,
            parent_channels=self.parent_channels,
            cluster_size=self.cluster_size,
            tau_c=self.tau_c
        )

    def __call__(
        self,
        red_child: jnp.ndarray,
        red_parent: jnp.ndarray,
        blue_child: jnp.ndarray,
        blue_parent: jnp.ndarray,
        key: jax.random.PRNGKey
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Execute one battle simulation step.

        Args:
            red_child: Red army child state
            red_parent: Red army parent state
            blue_child: Blue army child state
            blue_parent: Blue army parent state
            key: PRNG key

        Returns:
            Tuple of (new_red_child, new_red_parent, new_blue_child, new_blue_parent)
        """
        key1, key2 = jax.random.split(key)

        # Each army sees the other as enemy
        new_red_child, new_red_parent = self.red_hnca(
            red_child, red_parent, key1, enemy_state=blue_child
        )
        new_blue_child, new_blue_parent = self.blue_hnca(
            blue_child, blue_parent, key2, enemy_state=red_child
        )

        return new_red_child, new_red_parent, new_blue_child, new_blue_parent

    def simulate_battle(
        self,
        red_child: jnp.ndarray,
        red_parent: jnp.ndarray,
        blue_child: jnp.ndarray,
        blue_parent: jnp.ndarray,
        key: jax.random.PRNGKey,
        num_steps: int = 100
    ) -> dict[str, jnp.ndarray]:
        """Run full battle simulation.

        Args:
            red_child: Initial red child state
            red_parent: Initial red parent state
            blue_child: Initial blue child state
            blue_parent: Initial blue parent state
            key: PRNG key
            num_steps: Number of simulation steps

        Returns:
            Dictionary with 'red_child', 'red_parent', 'blue_child', 'blue_parent'
            trajectories
        """
        keys = jax.random.split(key, num_steps)

        def step_fn(carry, subkey):
            rc, rp, bc, bp = carry
            new_rc, new_rp, new_bc, new_bp = self(rc, rp, bc, bp, subkey)
            return (new_rc, new_rp, new_bc, new_bp), (new_rc, new_rp, new_bc, new_bp)

        _, trajectories = jax.lax.scan(
            step_fn,
            (red_child, red_parent, blue_child, blue_parent),
            keys
        )

        return {
            'red_child': trajectories[0],
            'red_parent': trajectories[1],
            'blue_child': trajectories[2],
            'blue_parent': trajectories[3]
        }


def create_battle_scenario(
    grid_size: int = 200,
    cluster_size: int = 4
) -> dict:
    """Create initial battle scenario with two opposing armies.

    Args:
        grid_size: Size of battle grid
        cluster_size: Cells per parent cluster

    Returns:
        Dictionary with 'red_child', 'red_parent', 'blue_child', 'blue_parent'
    """
    parent_size = grid_size // cluster_size

    # Red army: left side, line formation
    red_child = jnp.zeros((grid_size, grid_size, CHILD_CHANNELS.TOTAL))
    red_spawn = (grid_size // 2 - 10, grid_size // 2 + 10, 20, 50)
    y0, y1, x0, x1 = red_spawn

    red_child = red_child.at[y0:y1, x0:x1, 0].set(1.0)  # Red
    red_child = red_child.at[y0:y1, x0:x1, 3].set(1.0)  # Alpha
    red_child = red_child.at[y0:y1, x0:x1, 4].set(1.0)  # Health
    red_child = red_child.at[y0:y1, x0:x1, 5].set(0.5)  # Morale
    red_child = red_child.at[y0:y1, x0:x1, 15:].set(0.1)  # Hidden

    red_parent = jnp.zeros((parent_size, parent_size, PARENT_CHANNELS.TOTAL))
    red_parent = red_parent.at[..., 3].set(0.1)  # Low initial alpha
    red_parent = red_parent.at[..., 9:].set(0.1)  # Hidden

    # Blue army: right side, line formation
    blue_child = jnp.zeros((grid_size, grid_size, CHILD_CHANNELS.TOTAL))
    blue_spawn = (grid_size // 2 - 10, grid_size // 2 + 10, grid_size - 50, grid_size - 20)
    y0, y1, x0, x1 = blue_spawn

    blue_child = blue_child.at[y0:y1, x0:x1, 2].set(1.0)  # Blue
    blue_child = blue_child.at[y0:y1, x0:x1, 3].set(1.0)  # Alpha
    blue_child = blue_child.at[y0:y1, x0:x1, 4].set(1.0)  # Health
    blue_child = blue_child.at[y0:y1, x0:x1, 5].set(0.5)  # Morale
    blue_child = blue_child.at[y0:y1, x0:x1, 15:].set(0.1)  # Hidden

    blue_parent = jnp.zeros((parent_size, parent_size, PARENT_CHANNELS.TOTAL))
    blue_parent = blue_parent.at[..., 3].set(0.1)
    blue_parent = blue_parent.at[..., 9:].set(0.1)

    return {
        'red_child': red_child,
        'red_parent': red_parent,
        'blue_child': blue_child,
        'blue_parent': blue_parent
    }
