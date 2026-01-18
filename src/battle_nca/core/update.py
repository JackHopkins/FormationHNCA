"""Update rule networks for NCA."""

import jax.numpy as jnp
from flax import linen as nn
from typing import Callable


class NCAUpdateRule(nn.Module):
    """Neural network update rule for NCA.

    A small MLP that processes perception vectors and outputs residual
    state updates. The final layer is zero-initialized for stable
    "do-nothing" initial behavior.

    Attributes:
        num_channels: Number of output channels (state channels)
        hidden_dim: Hidden layer dimension
        num_hidden_layers: Number of hidden layers
        activation: Activation function
    """
    num_channels: int = 16
    hidden_dim: int = 128
    num_hidden_layers: int = 1
    activation: Callable = nn.relu

    @nn.compact
    def __call__(self, perception: jnp.ndarray) -> jnp.ndarray:
        """Compute residual state update from perception.

        Args:
            perception: Perception tensor of shape (..., H, W, P)

        Returns:
            Residual update tensor of shape (..., H, W, num_channels)
        """
        x = perception

        # Hidden layers
        for i in range(self.num_hidden_layers):
            x = nn.Conv(
                self.hidden_dim,
                kernel_size=(1, 1),
                name=f'hidden_{i}'
            )(x)
            x = self.activation(x)

        # Output layer with zero initialization for stable start
        ds = nn.Conv(
            self.num_channels,
            kernel_size=(1, 1),
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name='output'
        )(x)

        return ds


class BattleUpdateRule(nn.Module):
    """Specialized update rule for battle simulation.

    Includes separate pathways for different aspects of battle:
    - Combat updates (health, damage)
    - Morale updates (routing, rallying)
    - Movement updates (velocity, position)

    Attributes:
        num_channels: Number of output channels
        hidden_dim: Hidden layer dimension
    """
    num_channels: int = 24
    hidden_dim: int = 128

    @nn.compact
    def __call__(
        self,
        perception: jnp.ndarray,
        parent_signal: jnp.ndarray | None = None
    ) -> jnp.ndarray:
        """Compute battle state update.

        Args:
            perception: Perception tensor
            parent_signal: Optional parent NCA command signals

        Returns:
            Residual update tensor
        """
        x = perception

        # Incorporate parent signals if provided
        if parent_signal is not None:
            x = jnp.concatenate([x, parent_signal], axis=-1)

        # Shared hidden representation
        x = nn.Conv(self.hidden_dim, kernel_size=(1, 1), name='shared')(x)
        x = nn.relu(x)

        # Combat pathway (channels 4-6: health, morale, fatigue)
        combat_hidden = nn.Conv(64, kernel_size=(1, 1), name='combat_hidden')(x)
        combat_hidden = nn.relu(combat_hidden)
        combat_update = nn.Conv(
            3,
            kernel_size=(1, 1),
            kernel_init=nn.initializers.zeros,
            name='combat_out'
        )(combat_hidden)

        # Movement pathway (channels 7-8: velocity)
        movement_hidden = nn.Conv(32, kernel_size=(1, 1), name='movement_hidden')(x)
        movement_hidden = nn.relu(movement_hidden)
        movement_update = nn.Conv(
            2,
            kernel_size=(1, 1),
            kernel_init=nn.initializers.zeros,
            name='movement_out'
        )(movement_hidden)

        # Hidden state pathway (channels 15-23)
        hidden_update = nn.Conv(
            self.num_channels - 15,
            kernel_size=(1, 1),
            kernel_init=nn.initializers.zeros,
            name='hidden_out'
        )(x)

        # Other channels (0-3: RGBA, 9-14: type, formation, signals, enemy)
        other_update = nn.Conv(
            10,
            kernel_size=(1, 1),
            kernel_init=nn.initializers.zeros,
            name='other_out'
        )(x)

        # Combine all updates in channel order
        # Channels: 0-3 (RGBA), 4-6 (combat), 7-8 (movement), 9-14 (other), 15-23 (hidden)
        full_update = jnp.concatenate([
            other_update[..., :4],      # RGBA (0-3)
            combat_update,               # health, morale, fatigue (4-6)
            movement_update,             # velocity (7-8)
            other_update[..., 4:],       # type, formation, signals, enemy (9-14)
            hidden_update                # hidden state (15-23)
        ], axis=-1)

        return full_update


class GoalGuidedUpdateRule(nn.Module):
    """Update rule with goal conditioning for formation control.

    Uses a small encoder to map formation IDs to perturbation vectors
    that influence the update rule.

    Attributes:
        num_channels: Number of output channels
        hidden_dim: Hidden layer dimension
        num_formations: Number of possible formations
        goal_embed_dim: Dimension of goal embedding
    """
    num_channels: int = 24
    hidden_dim: int = 128
    num_formations: int = 5
    goal_embed_dim: int = 16

    @nn.compact
    def __call__(
        self,
        perception: jnp.ndarray,
        formation_id: int | jnp.ndarray
    ) -> jnp.ndarray:
        """Compute goal-conditioned state update.

        Args:
            perception: Perception tensor
            formation_id: Target formation index

        Returns:
            Residual update tensor
        """
        # Embed formation goal
        goal_embed = nn.Embed(
            num_embeddings=self.num_formations,
            features=self.goal_embed_dim,
            name='goal_embed'
        )(formation_id)

        # Broadcast goal to spatial dimensions
        spatial_shape = perception.shape[:-1]
        goal_broadcast = jnp.broadcast_to(
            goal_embed,
            (*spatial_shape, self.goal_embed_dim)
        )

        # Concatenate perception with goal
        x = jnp.concatenate([perception, goal_broadcast], axis=-1)

        # Standard update computation
        x = nn.Conv(self.hidden_dim, kernel_size=(1, 1))(x)
        x = nn.relu(x)

        ds = nn.Conv(
            self.num_channels,
            kernel_size=(1, 1),
            kernel_init=nn.initializers.zeros
        )(x)

        return ds
