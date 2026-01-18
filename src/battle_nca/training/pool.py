"""Sample pool for stable NCA training."""

import jax
import jax.numpy as jnp
from typing import NamedTuple
from functools import partial


class PoolState(NamedTuple):
    """State of the sample pool."""
    samples: jnp.ndarray  # (pool_size, H, W, C)
    seed: jnp.ndarray     # (H, W, C)


class NCAPool:
    """Sample pool for stable NCA training.

    Pool-based training is critical for NCA stability. Rather than
    backpropagating through thousands of timesteps, maintain a pool
    of intermediate states. Sample batches from the pool, train,
    then inject outputs back.

    This creates attractor dynamics: the NCA learns not just trajectories
    to targets, but how to persist at and return to targets.

    Attributes:
        pool_size: Number of samples in pool
        seed: Initial seed state
    """

    def __init__(
        self,
        seed: jnp.ndarray,
        pool_size: int = 1024
    ):
        """Initialize pool with copies of seed.

        Args:
            seed: Seed state (H, W, C)
            pool_size: Number of samples in pool
        """
        self.pool_size = pool_size
        self.seed = seed
        self.samples = jnp.tile(seed[None], (pool_size, 1, 1, 1))

    def sample(
        self,
        batch_size: int,
        key: jax.random.PRNGKey
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Sample a batch from the pool.

        Args:
            batch_size: Number of samples
            key: PRNG key

        Returns:
            Tuple of (indices, batch)
        """
        indices = jax.random.choice(
            key, self.pool_size, shape=(batch_size,), replace=False
        )
        batch = self.samples[indices]
        return indices, batch

    def update(
        self,
        indices: jnp.ndarray,
        outputs: jnp.ndarray,
        losses: jnp.ndarray
    ) -> None:
        """Update pool with new outputs, replacing highest-loss with seed.

        Args:
            indices: Indices of sampled items
            outputs: New states from training
            losses: Per-sample losses
        """
        # Sort by loss descending
        sorted_order = jnp.argsort(-losses)
        sorted_indices = indices[sorted_order]
        sorted_outputs = outputs[sorted_order]

        # Replace highest-loss sample with seed (prevents forgetting)
        sorted_outputs = sorted_outputs.at[0].set(self.seed)

        # Update pool
        self.samples = self.samples.at[sorted_indices].set(sorted_outputs)

    def apply_damage(
        self,
        batch: jnp.ndarray,
        num_damage: int,
        key: jax.random.PRNGKey,
        min_radius: int = 5,
        max_radius: int = 15
    ) -> jnp.ndarray:
        """Apply circular damage to lowest-loss samples.

        Damage augmentation trains regeneration capability.

        Args:
            batch: Batch of samples (B, H, W, C)
            num_damage: Number of samples to damage
            key: PRNG key
            min_radius: Minimum damage radius
            max_radius: Maximum damage radius

        Returns:
            Batch with damage applied to first num_damage samples
        """
        batch_size, height, width, channels = batch.shape
        num_damage = min(num_damage, batch_size)

        # Create coordinate grids
        y_coords = jnp.arange(height)
        x_coords = jnp.arange(width)
        yy, xx = jnp.meshgrid(y_coords, x_coords, indexing='ij')

        for i in range(num_damage):
            key, subkey1, subkey2 = jax.random.split(key, 3)

            # Random center (avoiding edges)
            margin = max_radius
            cy = jax.random.randint(subkey1, (), margin, height - margin)
            cx = jax.random.randint(subkey1, (), margin, width - margin)

            # Random radius
            radius = jax.random.randint(subkey2, (), min_radius, max_radius + 1)

            # Create circular mask
            dist_sq = (yy - cy) ** 2 + (xx - cx) ** 2
            mask = dist_sq <= radius ** 2

            # Zero out damaged region
            mask_expanded = mask[..., None]
            batch = batch.at[i].set(
                jnp.where(mask_expanded, 0.0, batch[i])
            )

        return batch

    def get_state(self) -> PoolState:
        """Get pool state for checkpointing."""
        return PoolState(samples=self.samples, seed=self.seed)

    def set_state(self, state: PoolState) -> None:
        """Restore pool state from checkpoint."""
        self.samples = state.samples
        self.seed = state.seed

    @property
    def mean_alpha(self) -> float:
        """Get mean alpha across pool (measure of activity)."""
        return float(jnp.mean(self.samples[..., 3]))


class HierarchicalPool:
    """Pool for hierarchical NCA training.

    Manages paired child/parent state pools.
    """

    def __init__(
        self,
        child_seed: jnp.ndarray,
        parent_seed: jnp.ndarray,
        pool_size: int = 1024
    ):
        """Initialize hierarchical pool.

        Args:
            child_seed: Child NCA seed
            parent_seed: Parent NCA seed
            pool_size: Pool size
        """
        self.pool_size = pool_size
        self.child_seed = child_seed
        self.parent_seed = parent_seed

        self.child_samples = jnp.tile(child_seed[None], (pool_size, 1, 1, 1))
        self.parent_samples = jnp.tile(parent_seed[None], (pool_size, 1, 1, 1))

    def sample(
        self,
        batch_size: int,
        key: jax.random.PRNGKey
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Sample paired child/parent states.

        Args:
            batch_size: Number of samples
            key: PRNG key

        Returns:
            Tuple of (indices, child_batch, parent_batch)
        """
        indices = jax.random.choice(
            key, self.pool_size, shape=(batch_size,), replace=False
        )
        child_batch = self.child_samples[indices]
        parent_batch = self.parent_samples[indices]
        return indices, child_batch, parent_batch

    def update(
        self,
        indices: jnp.ndarray,
        child_outputs: jnp.ndarray,
        parent_outputs: jnp.ndarray,
        losses: jnp.ndarray
    ) -> None:
        """Update both pools with new outputs.

        Args:
            indices: Sampled indices
            child_outputs: New child states
            parent_outputs: New parent states
            losses: Per-sample losses
        """
        sorted_order = jnp.argsort(-losses)
        sorted_indices = indices[sorted_order]
        sorted_child = child_outputs[sorted_order]
        sorted_parent = parent_outputs[sorted_order]

        # Replace worst with seeds
        sorted_child = sorted_child.at[0].set(self.child_seed)
        sorted_parent = sorted_parent.at[0].set(self.parent_seed)

        self.child_samples = self.child_samples.at[sorted_indices].set(sorted_child)
        self.parent_samples = self.parent_samples.at[sorted_indices].set(sorted_parent)

    def apply_child_damage(
        self,
        child_batch: jnp.ndarray,
        num_damage: int,
        key: jax.random.PRNGKey
    ) -> jnp.ndarray:
        """Apply damage to child states for regeneration training."""
        pool = NCAPool(self.child_seed, self.pool_size)
        return pool.apply_damage(child_batch, num_damage, key)


@partial(jax.jit, static_argnums=(1, 2, 5, 6))
def jit_sample_and_damage(
    pool_samples: jnp.ndarray,
    batch_size: int,
    pool_size: int,
    seed: jnp.ndarray,
    key: jax.random.PRNGKey,
    num_damage: int = 3,
    apply_damage: bool = True
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """JIT-compiled pool sampling with optional damage.

    Args:
        pool_samples: Pool samples array
        batch_size: Batch size
        pool_size: Total pool size
        seed: Seed state
        key: PRNG key
        num_damage: Number of samples to damage
        apply_damage: Whether to apply damage

    Returns:
        Tuple of (indices, batch)
    """
    key1, key2 = jax.random.split(key)

    # Sample
    indices = jax.random.choice(
        key1, pool_size, shape=(batch_size,), replace=False
    )
    batch = pool_samples[indices]

    if apply_damage and num_damage > 0:
        # Simple damage implementation for JIT
        height, width = batch.shape[1:3]
        margin = 15

        for i in range(min(num_damage, batch_size)):
            key2, subkey = jax.random.split(key2)
            cy = jax.random.randint(subkey, (), margin, height - margin)
            key2, subkey = jax.random.split(key2)
            cx = jax.random.randint(subkey, (), margin, width - margin)
            key2, subkey = jax.random.split(key2)
            radius = jax.random.randint(subkey, (), 5, 15)

            y_coords = jnp.arange(height)
            x_coords = jnp.arange(width)
            yy, xx = jnp.meshgrid(y_coords, x_coords, indexing='ij')
            mask = ((yy - cy) ** 2 + (xx - cx) ** 2) <= radius ** 2

            batch = batch.at[i].set(
                jnp.where(mask[..., None], 0.0, batch[i])
            )

    return indices, batch
