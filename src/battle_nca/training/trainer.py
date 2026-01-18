"""Training loop and curriculum for battle NCA."""

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
from dataclasses import dataclass, field
from typing import Any, Callable
from functools import partial
import time

from battle_nca.training.pool import NCAPool, HierarchicalPool
from battle_nca.training.optimizers import create_optimizer, normalize_gradients
from battle_nca.combat.losses import formation_loss, total_battle_loss
from battle_nca.combat.formations import FormationTypes, create_formation_target


@dataclass
class TrainingConfig:
    """Configuration for NCA training.

    Attributes:
        batch_size: Samples per batch
        pool_size: Total pool size
        min_steps: Minimum steps per sample
        max_steps: Maximum steps per sample
        learning_rate: Base learning rate
        gradient_clip: Gradient clipping threshold
        damage_samples: Number of samples to damage per batch
        damage_start_epoch: Epoch to start damage augmentation
        log_interval: Epochs between logging
        checkpoint_interval: Epochs between checkpoints
    """
    batch_size: int = 32
    pool_size: int = 1024
    min_steps: int = 64
    max_steps: int = 96
    learning_rate: float = 2e-3
    gradient_clip: float = 1.0
    damage_samples: int = 3
    damage_start_epoch: int = 1000
    log_interval: int = 100
    checkpoint_interval: int = 500


@dataclass
class TrainingState:
    """State of training run."""
    epoch: int = 0
    best_loss: float = float('inf')
    losses: list = field(default_factory=list)
    times: list = field(default_factory=list)


class Trainer:
    """Trainer for battle NCA with curriculum learning.

    Implements three-phase curriculum:
    1. Static formation learning (1,500 iterations)
    2. Multi-formation transitions (2,500 iterations)
    3. Combat dynamics (4,000 iterations)
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig | None = None,
        seed: jnp.ndarray | None = None
    ):
        """Initialize trainer.

        Args:
            model: NCA model to train
            config: Training configuration
            seed: Initial seed state
        """
        self.model = model
        self.config = config or TrainingConfig()
        self.seed = seed
        self.state = TrainingState()

    def create_train_state(
        self,
        key: jax.random.PRNGKey,
        dummy_input: jnp.ndarray
    ) -> train_state.TrainState:
        """Create Flax training state.

        Args:
            key: PRNG key for initialization
            dummy_input: Dummy input for parameter initialization

        Returns:
            Flax TrainState
        """
        variables = self.model.init(key, dummy_input, jax.random.PRNGKey(0))
        params = variables['params']

        optimizer = create_optimizer(
            learning_rate=self.config.learning_rate,
            gradient_clip=self.config.gradient_clip
        )

        return train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=params,
            tx=optimizer
        )

    @partial(jax.jit, static_argnums=(0,))
    def _train_step(
        self,
        state: train_state.TrainState,
        batch: jnp.ndarray,
        target: jnp.ndarray,
        key: jax.random.PRNGKey,
        num_steps: int
    ) -> tuple[train_state.TrainState, jnp.ndarray, jnp.ndarray]:
        """Single JIT-compiled training step.

        Args:
            state: Current training state
            batch: Batch of samples
            target: Target formation
            key: PRNG key
            num_steps: Steps to run

        Returns:
            Tuple of (new_state, loss, outputs)
        """
        def loss_fn(params):
            keys = jax.random.split(key, num_steps)

            def step(carry, subkey):
                return self.model.apply({'params': params}, carry, subkey), None

            final, _ = jax.lax.scan(step, batch, keys)

            loss = formation_loss(final, target)
            return loss, final

        (loss, outputs), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        grads = normalize_gradients(grads)
        state = state.apply_gradients(grads=grads)

        return state, loss, outputs

    def train_phase1(
        self,
        train_state: train_state.TrainState,
        target: jnp.ndarray,
        key: jax.random.PRNGKey,
        num_epochs: int = 1500,
        pool: NCAPool | None = None
    ) -> tuple[train_state.TrainState, dict]:
        """Phase 1: Static formation learning.

        Args:
            train_state: Initial training state
            target: Target formation
            key: PRNG key
            num_epochs: Number of training epochs
            pool: Optional pre-existing pool

        Returns:
            Tuple of (trained_state, metrics)
        """
        if pool is None:
            pool = NCAPool(self.seed, self.config.pool_size)

        metrics = {'losses': [], 'times': []}

        for epoch in range(num_epochs):
            start_time = time.time()

            key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)

            # Sample from pool
            indices, batch = pool.sample(self.config.batch_size, subkey1)

            # Apply damage after warmup
            if epoch > self.config.damage_start_epoch:
                batch = pool.apply_damage(batch, self.config.damage_samples, subkey2)

            # Random step count
            num_steps = jax.random.randint(
                subkey3, (), self.config.min_steps, self.config.max_steps + 1
            )

            # Train step
            train_state, loss, outputs = self._train_step(
                train_state, batch, target, subkey3, num_steps
            )

            # Compute per-sample losses for pool update
            per_sample_losses = jax.vmap(
                lambda s: jnp.mean((s[..., :4] - target) ** 2)
            )(outputs)

            # Update pool
            pool.update(indices, outputs, per_sample_losses)

            elapsed = time.time() - start_time
            metrics['losses'].append(float(loss))
            metrics['times'].append(elapsed)

            if epoch % self.config.log_interval == 0:
                print(f"Phase 1 | Epoch {epoch}: loss = {loss:.6f}, time = {elapsed:.3f}s")

        return train_state, metrics

    def train_phase2(
        self,
        train_state: train_state.TrainState,
        key: jax.random.PRNGKey,
        num_epochs: int = 2500,
        height: int = 64,
        width: int = 64
    ) -> tuple[train_state.TrainState, dict]:
        """Phase 2: Multi-formation transitions.

        Trains on random formation switches to learn goal-conditioned behavior.

        Args:
            train_state: Initial training state
            key: PRNG key
            num_epochs: Number of training epochs
            height: Grid height
            width: Grid width

        Returns:
            Tuple of (trained_state, metrics)
        """
        formations = list(FormationTypes)
        targets = {ft: create_formation_target(height, width, ft) for ft in formations}

        pool = NCAPool(self.seed, self.config.pool_size)
        metrics = {'losses': [], 'times': []}

        for epoch in range(num_epochs):
            start_time = time.time()

            key, subkey1, subkey2, subkey3, subkey4 = jax.random.split(key, 5)

            # Sample random target formation
            target_idx = jax.random.randint(subkey1, (), 0, len(formations))
            target = targets[formations[target_idx]]

            # Sample from pool
            indices, batch = pool.sample(self.config.batch_size, subkey2)

            # Damage augmentation
            if epoch > self.config.damage_start_epoch:
                batch = pool.apply_damage(batch, self.config.damage_samples, subkey3)

            # Random steps
            num_steps = jax.random.randint(
                subkey4, (), self.config.min_steps, self.config.max_steps + 1
            )

            # Train
            train_state, loss, outputs = self._train_step(
                train_state, batch, target, subkey4, num_steps
            )

            # Update pool
            per_sample_losses = jax.vmap(
                lambda s: jnp.mean((s[..., :4] - target) ** 2)
            )(outputs)
            pool.update(indices, outputs, per_sample_losses)

            elapsed = time.time() - start_time
            metrics['losses'].append(float(loss))
            metrics['times'].append(elapsed)

            if epoch % self.config.log_interval == 0:
                print(f"Phase 2 | Epoch {epoch}: loss = {loss:.6f}, "
                      f"target = {formations[target_idx].name}")

        return train_state, metrics

    def train_phase3(
        self,
        child_state: train_state.TrainState,
        parent_state: train_state.TrainState,
        key: jax.random.PRNGKey,
        num_epochs: int = 4000,
        freeze_child: bool = True
    ) -> tuple[train_state.TrainState, train_state.TrainState, dict]:
        """Phase 3: Combat dynamics with hierarchical training.

        Trains parent-NCA to coordinate child formations in adversarial setting.

        Args:
            child_state: Child NCA training state
            parent_state: Parent NCA training state
            key: PRNG key
            num_epochs: Number of training epochs
            freeze_child: Whether to freeze child weights

        Returns:
            Tuple of (child_state, parent_state, metrics)
        """
        # This phase requires the full HierarchicalNCA - placeholder implementation
        metrics = {'losses': [], 'times': []}

        print("Phase 3: Combat dynamics training")
        print("Note: Full implementation requires HierarchicalNCA adversarial training")

        # Placeholder: return unchanged states
        return child_state, parent_state, metrics

    def full_curriculum(
        self,
        key: jax.random.PRNGKey,
        target: jnp.ndarray,
        height: int = 64,
        width: int = 64,
        phase1_epochs: int = 1500,
        phase2_epochs: int = 2500,
        phase3_epochs: int = 4000
    ) -> dict:
        """Run full three-phase curriculum.

        Args:
            key: PRNG key
            target: Initial target formation
            height: Grid height
            width: Grid width
            phase1_epochs: Phase 1 epochs
            phase2_epochs: Phase 2 epochs
            phase3_epochs: Phase 3 epochs

        Returns:
            Dictionary with final states and metrics
        """
        key1, key2, key3, key4 = jax.random.split(key, 4)

        # Initialize
        dummy_input = jnp.zeros((height, width, self.model.num_channels))
        train_state = self.create_train_state(key1, dummy_input)

        print("=" * 60)
        print("Starting Phase 1: Static Formation Learning")
        print("=" * 60)
        train_state, phase1_metrics = self.train_phase1(
            train_state, target, key2, phase1_epochs
        )

        print("\n" + "=" * 60)
        print("Starting Phase 2: Multi-Formation Transitions")
        print("=" * 60)
        train_state, phase2_metrics = self.train_phase2(
            train_state, key3, phase2_epochs, height, width
        )

        print("\n" + "=" * 60)
        print("Starting Phase 3: Combat Dynamics")
        print("=" * 60)
        # Phase 3 requires hierarchical model
        phase3_metrics = {'losses': [], 'times': []}

        return {
            'train_state': train_state,
            'phase1_metrics': phase1_metrics,
            'phase2_metrics': phase2_metrics,
            'phase3_metrics': phase3_metrics
        }


def create_trainer(
    model: nn.Module,
    seed: jnp.ndarray,
    config: TrainingConfig | None = None
) -> Trainer:
    """Factory function to create trainer.

    Args:
        model: NCA model
        seed: Initial seed state
        config: Optional training config

    Returns:
        Configured Trainer
    """
    return Trainer(model=model, config=config, seed=seed)


@partial(jax.jit, static_argnums=(4,))
def train_step_jit(
    params: Any,
    batch: jnp.ndarray,
    target: jnp.ndarray,
    key: jax.random.PRNGKey,
    apply_fn: Callable,
    num_steps: int
) -> tuple[Any, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Standalone JIT-compiled training step.

    For use outside the Trainer class.

    Args:
        params: Model parameters
        batch: Input batch
        target: Target formation
        key: PRNG key
        apply_fn: Model apply function
        num_steps: Number of NCA steps

    Returns:
        Tuple of (gradients, loss, outputs, per_sample_losses)
    """
    def loss_fn(p):
        keys = jax.random.split(key, num_steps)

        def step(carry, subkey):
            return apply_fn({'params': p}, carry, subkey), None

        final, _ = jax.lax.scan(step, batch, keys)
        loss = jnp.mean((final[..., :4] - target) ** 2)
        return loss, final

    (loss, outputs), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    grads = normalize_gradients(grads)

    per_sample_losses = jax.vmap(
        lambda s: jnp.mean((s[..., :4] - target) ** 2)
    )(outputs)

    return grads, loss, outputs, per_sample_losses
