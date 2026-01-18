"""Optimizers and gradient utilities for NCA training."""

import jax
import jax.numpy as jnp
import optax
from typing import Any

PyTree = Any


def normalize_gradients(grads: PyTree) -> PyTree:
    """Normalize gradients per-variable to unit norm.

    This is the Growing NCA approach to gradient stabilization,
    preventing any single variable from dominating the update.

    Args:
        grads: PyTree of gradients

    Returns:
        Normalized gradients
    """
    def norm_grad(g: jnp.ndarray) -> jnp.ndarray:
        # Replace NaN/inf with zeros
        g = jnp.where(jnp.isfinite(g), g, 0.0)
        norm = jnp.linalg.norm(g)
        # Only normalize if norm is reasonable
        return jnp.where(norm > 1e-8, g / norm, g * 0.0)

    return jax.tree.map(norm_grad, grads)


def create_optimizer(
    learning_rate: float = 2e-3,
    gradient_clip: float = 1.0,
    use_schedule: bool = True,
    warmup_steps: int = 500,
    decay_steps: int = 8000
) -> optax.GradientTransformation:
    """Create optimizer with gradient clipping and optional schedule.

    Args:
        learning_rate: Base learning rate
        gradient_clip: Maximum gradient norm
        use_schedule: Whether to use learning rate schedule
        warmup_steps: Warmup steps for schedule
        decay_steps: Total decay steps for schedule

    Returns:
        Optax optimizer
    """
    if use_schedule:
        # Warmup + cosine decay schedule
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=learning_rate * 0.1,
            peak_value=learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            end_value=learning_rate * 0.01
        )
    else:
        schedule = learning_rate

    return optax.chain(
        optax.clip_by_global_norm(gradient_clip),
        optax.adam(schedule)
    )


def create_nca_optimizer(
    learning_rate: float = 2e-3,
    use_per_variable_norm: bool = True
) -> optax.GradientTransformation:
    """Create NCA-specific optimizer.

    Combines gradient clipping with optional per-variable normalization
    as used in Growing NCA.

    Args:
        learning_rate: Learning rate
        use_per_variable_norm: Whether to normalize per variable

    Returns:
        Optax optimizer
    """
    transforms = [optax.clip_by_global_norm(1.0)]

    if use_per_variable_norm:
        # Custom transform for per-variable normalization
        def per_var_norm(updates, state, params=None):
            del state, params
            return normalize_gradients(updates), optax.EmptyState()

        transforms.append(optax.stateless(per_var_norm))

    transforms.append(optax.adam(learning_rate))

    return optax.chain(*transforms)


def create_curriculum_optimizer(
    phase: int,
    base_lr: float = 2e-3
) -> optax.GradientTransformation:
    """Create optimizer for specific training phase.

    Args:
        phase: Training phase (1, 2, or 3)
        base_lr: Base learning rate

    Returns:
        Optax optimizer
    """
    # Phase-specific learning rates
    phase_lrs = {
        1: base_lr,           # Formation learning
        2: base_lr * 0.5,     # Multi-formation
        3: base_lr * 0.25     # Combat dynamics
    }

    lr = phase_lrs.get(phase, base_lr)

    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(lr)
    )


class AdaptiveGradientNormalizer:
    """Adaptive gradient normalization based on training progress.

    Starts with strong normalization and relaxes over time.
    """

    def __init__(self, initial_strength: float = 1.0, decay_rate: float = 0.999):
        self.strength = initial_strength
        self.decay_rate = decay_rate
        self.step = 0

    def __call__(self, grads: PyTree) -> PyTree:
        """Apply adaptive normalization.

        Args:
            grads: Input gradients

        Returns:
            Normalized gradients
        """
        # Compute current strength
        current_strength = self.strength * (self.decay_rate ** self.step)
        self.step += 1

        def adaptive_norm(g: jnp.ndarray) -> jnp.ndarray:
            norm = jnp.linalg.norm(g)
            normalized = g / (norm + 1e-8)
            # Interpolate between normalized and original
            return current_strength * normalized + (1 - current_strength) * g

        return jax.tree.map(adaptive_norm, grads)


def compute_gradient_stats(grads: PyTree) -> dict[str, float]:
    """Compute gradient statistics for monitoring.

    Args:
        grads: Gradient PyTree

    Returns:
        Dictionary of statistics
    """
    flat_grads = jax.tree_util.tree_leaves(grads)

    norms = [jnp.linalg.norm(g) for g in flat_grads]
    total_norm = jnp.sqrt(sum(n ** 2 for n in norms))

    return {
        'grad_norm': float(total_norm),
        'grad_max': float(max(jnp.max(jnp.abs(g)) for g in flat_grads)),
        'grad_min': float(min(jnp.min(jnp.abs(g)) for g in flat_grads)),
        'num_params': sum(g.size for g in flat_grads)
    }
