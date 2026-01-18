"""Evaluation metrics for battle NCA."""

import jax
import jax.numpy as jnp
import numpy as np
from typing import NamedTuple
from dataclasses import dataclass

from battle_nca.combat.channels import Channels

CH = Channels()


@dataclass
class ArmyStatistics:
    """Statistics for an army."""
    total_units: int
    alive_units: int
    average_health: float
    average_morale: float
    average_fatigue: float
    routing_units: int
    unit_density: float


@dataclass
class BattleMetrics:
    """Metrics for a battle state."""
    red_stats: ArmyStatistics
    blue_stats: ArmyStatistics
    red_advantage: float  # Positive = red winning
    battle_intensity: float  # Combat activity level
    formation_quality_red: float
    formation_quality_blue: float


def compute_army_statistics(
    state: jnp.ndarray,
    routing_threshold: float = -0.5
) -> ArmyStatistics:
    """Compute statistics for an army.

    Args:
        state: Army state (H, W, C)
        routing_threshold: Morale below this = routing

    Returns:
        ArmyStatistics
    """
    state = np.array(state)

    alpha = state[..., CH.ALPHA]
    health = state[..., CH.HEALTH]
    morale = state[..., CH.MORALE]
    fatigue = state[..., CH.FATIGUE]

    alive_mask = alpha > 0.1
    total_cells = state.shape[0] * state.shape[1]

    total_units = int(np.sum(alpha > 0.5))
    alive_units = int(np.sum(alive_mask))

    if alive_units > 0:
        avg_health = float(np.mean(health[alive_mask]))
        avg_morale = float(np.mean(morale[alive_mask]))
        avg_fatigue = float(np.mean(fatigue[alive_mask]))
    else:
        avg_health = 0.0
        avg_morale = 0.0
        avg_fatigue = 0.0

    routing_mask = alive_mask & (morale < routing_threshold)
    routing_units = int(np.sum(routing_mask))

    unit_density = alive_units / total_cells

    return ArmyStatistics(
        total_units=total_units,
        alive_units=alive_units,
        average_health=avg_health,
        average_morale=avg_morale,
        average_fatigue=avg_fatigue,
        routing_units=routing_units,
        unit_density=unit_density
    )


def compute_battle_metrics(
    red_state: jnp.ndarray,
    blue_state: jnp.ndarray,
    red_target: jnp.ndarray | None = None,
    blue_target: jnp.ndarray | None = None
) -> BattleMetrics:
    """Compute metrics for a battle.

    Args:
        red_state: Red army state
        blue_state: Blue army state
        red_target: Optional red formation target
        blue_target: Optional blue formation target

    Returns:
        BattleMetrics
    """
    red_stats = compute_army_statistics(red_state)
    blue_stats = compute_army_statistics(blue_state)

    # Red advantage (positive = red winning)
    total_red = max(red_stats.alive_units, 1)
    total_blue = max(blue_stats.alive_units, 1)
    red_advantage = (total_red - total_blue) / (total_red + total_blue)

    # Battle intensity (how much combat is happening)
    red_state_np = np.array(red_state)
    blue_state_np = np.array(blue_state)

    red_alpha = red_state_np[..., CH.ALPHA]
    blue_alpha = blue_state_np[..., CH.ALPHA]

    # Approximate combat activity from health changes / proximity
    contact = red_alpha * blue_alpha  # Where both armies present
    battle_intensity = float(np.mean(contact))

    # Formation quality
    if red_target is not None:
        red_target_np = np.array(red_target)
        red_quality = 1.0 - float(np.mean(
            (red_state_np[..., :4] - red_target_np) ** 2
        ))
    else:
        red_quality = 0.0

    if blue_target is not None:
        blue_target_np = np.array(blue_target)
        blue_quality = 1.0 - float(np.mean(
            (blue_state_np[..., :4] - blue_target_np) ** 2
        ))
    else:
        blue_quality = 0.0

    return BattleMetrics(
        red_stats=red_stats,
        blue_stats=blue_stats,
        red_advantage=red_advantage,
        battle_intensity=battle_intensity,
        formation_quality_red=red_quality,
        formation_quality_blue=blue_quality
    )


def compute_formation_metrics(
    state: jnp.ndarray,
    target: jnp.ndarray
) -> dict[str, float]:
    """Compute formation quality metrics.

    Args:
        state: Current NCA state
        target: Target formation

    Returns:
        Dictionary of metrics
    """
    state = np.array(state)
    target = np.array(target)

    state_alpha = state[..., CH.ALPHA]
    target_alpha = target[..., CH.ALPHA] if target.shape[-1] > 3 else target[..., 0]

    # MSE
    mse = float(np.mean((state_alpha - target_alpha) ** 2))

    # IoU
    threshold = 0.5
    state_mask = state_alpha > threshold
    target_mask = target_alpha > threshold
    intersection = np.sum(state_mask & target_mask)
    union = np.sum(state_mask | target_mask)
    iou = float(intersection / (union + 1e-6))

    # Coverage
    coverage = float(np.sum(state_mask & target_mask) / (np.sum(target_mask) + 1e-6))

    # Precision
    precision = float(np.sum(state_mask & target_mask) / (np.sum(state_mask) + 1e-6))

    # F1 score
    f1 = 2 * precision * coverage / (precision + coverage + 1e-6)

    return {
        'mse': mse,
        'iou': iou,
        'coverage': coverage,
        'precision': precision,
        'f1': f1
    }


def compute_trajectory_metrics(
    trajectory: jnp.ndarray,
    target: jnp.ndarray
) -> dict[str, np.ndarray]:
    """Compute metrics over a trajectory.

    Args:
        trajectory: State trajectory (T, H, W, C)
        target: Target formation

    Returns:
        Dictionary of metric arrays over time
    """
    trajectory = np.array(trajectory)
    target = np.array(target)
    num_steps = len(trajectory)

    metrics = {
        'mse': np.zeros(num_steps),
        'iou': np.zeros(num_steps),
        'alive_cells': np.zeros(num_steps),
        'mean_health': np.zeros(num_steps),
        'mean_morale': np.zeros(num_steps)
    }

    for t in range(num_steps):
        state = trajectory[t]

        # Formation metrics
        form_metrics = compute_formation_metrics(state, target)
        metrics['mse'][t] = form_metrics['mse']
        metrics['iou'][t] = form_metrics['iou']

        # Army stats
        stats = compute_army_statistics(state)
        metrics['alive_cells'][t] = stats.alive_units
        metrics['mean_health'][t] = stats.average_health
        metrics['mean_morale'][t] = stats.average_morale

    return metrics


def compute_stability_metrics(
    trajectory: jnp.ndarray,
    target: jnp.ndarray,
    window: int = 20
) -> dict[str, float]:
    """Compute stability metrics from trajectory.

    Measures how stable the NCA is at maintaining the target pattern.

    Args:
        trajectory: State trajectory (T, H, W, C)
        target: Target formation
        window: Number of final steps to analyze

    Returns:
        Stability metrics
    """
    trajectory = np.array(trajectory)
    target = np.array(target)

    # Get last N steps
    final_trajectory = trajectory[-window:]

    # Compute variance in alpha channel over final steps
    alpha_trajectory = final_trajectory[..., CH.ALPHA]
    temporal_variance = float(np.mean(np.var(alpha_trajectory, axis=0)))

    # Mean loss over final steps
    target_alpha = target[..., CH.ALPHA] if target.shape[-1] > 3 else target[..., 0]
    final_losses = [
        np.mean((s[..., CH.ALPHA] - target_alpha) ** 2)
        for s in final_trajectory
    ]
    mean_final_loss = float(np.mean(final_losses))
    loss_variance = float(np.var(final_losses))

    # Convergence rate (how quickly loss decreases)
    full_losses = [
        np.mean((s[..., CH.ALPHA] - target_alpha) ** 2)
        for s in trajectory
    ]
    if len(full_losses) > 10:
        early_loss = np.mean(full_losses[:10])
        late_loss = np.mean(full_losses[-10:])
        convergence_ratio = late_loss / (early_loss + 1e-6)
    else:
        convergence_ratio = 1.0

    return {
        'temporal_variance': temporal_variance,
        'mean_final_loss': mean_final_loss,
        'loss_variance': loss_variance,
        'convergence_ratio': convergence_ratio
    }


def print_battle_summary(metrics: BattleMetrics) -> None:
    """Print formatted battle summary.

    Args:
        metrics: Battle metrics to display
    """
    print("=" * 50)
    print("BATTLE SUMMARY")
    print("=" * 50)

    print("\nRED ARMY:")
    print(f"  Alive units: {metrics.red_stats.alive_units}")
    print(f"  Average health: {metrics.red_stats.average_health:.2%}")
    print(f"  Average morale: {metrics.red_stats.average_morale:+.2f}")
    print(f"  Routing units: {metrics.red_stats.routing_units}")
    print(f"  Formation quality: {metrics.formation_quality_red:.2%}")

    print("\nBLUE ARMY:")
    print(f"  Alive units: {metrics.blue_stats.alive_units}")
    print(f"  Average health: {metrics.blue_stats.average_health:.2%}")
    print(f"  Average morale: {metrics.blue_stats.average_morale:+.2f}")
    print(f"  Routing units: {metrics.blue_stats.routing_units}")
    print(f"  Formation quality: {metrics.formation_quality_blue:.2%}")

    print("\nBATTLE STATUS:")
    if metrics.red_advantage > 0.1:
        print(f"  Red is WINNING (advantage: {metrics.red_advantage:+.2f})")
    elif metrics.red_advantage < -0.1:
        print(f"  Blue is WINNING (advantage: {-metrics.red_advantage:+.2f})")
    else:
        print(f"  Battle is CONTESTED (advantage: {metrics.red_advantage:+.2f})")
    print(f"  Battle intensity: {metrics.battle_intensity:.2%}")
    print("=" * 50)
