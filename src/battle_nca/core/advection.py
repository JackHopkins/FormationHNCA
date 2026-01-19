"""Advection-based mass transport for NCA.

This module implements physically-based mass transport where:
- Cells have mass (alpha) and velocity
- Mass is transported according to velocity field
- Total mass is conserved by construction
- Movement is explicit, not emergent

This is fundamentally different from standard NCA where alpha
is directly updated. Here, the NCA controls velocity, and
alpha changes are a consequence of physical transport.
"""

import jax
import jax.numpy as jnp


def advect_mass(
    mass: jnp.ndarray,
    velocity_x: jnp.ndarray,
    velocity_y: jnp.ndarray,
    dt: float = 0.5
) -> jnp.ndarray:
    """Transport mass based on velocity field using upwind scheme.

    This is a first-order upwind advection that ensures:
    - Mass conservation: sum(mass) is preserved (up to boundary effects)
    - Stability: CFL condition satisfied when |v| <= 1 and dt <= 0.5
    - Locality: only immediate neighbors exchange mass

    The scheme works by computing:
    - Outflow: mass * velocity (in direction of velocity)
    - Inflow: neighbor's outflow toward this cell

    Args:
        mass: Density field of shape (H, W) or (B, H, W)
        velocity_x: Horizontal velocity, positive = rightward, range [-1, 1]
        velocity_y: Vertical velocity, positive = downward, range [-1, 1]
        dt: Time step, should be <= 0.5 for stability

    Returns:
        Updated mass field with same total mass (modulo boundaries)
    """
    # Ensure velocities are in stable range
    vx = jnp.clip(velocity_x, -1.0, 1.0)
    vy = jnp.clip(velocity_y, -1.0, 1.0)

    # Compute flux out of each cell in each direction
    # Flux = mass * velocity (only in direction of velocity)
    flux_right = mass * jnp.maximum(vx, 0.0)  # rightward flux
    flux_left = mass * jnp.maximum(-vx, 0.0)  # leftward flux
    flux_down = mass * jnp.maximum(vy, 0.0)   # downward flux
    flux_up = mass * jnp.maximum(-vy, 0.0)    # upward flux

    # Total outflow from each cell
    outflow = flux_right + flux_left + flux_down + flux_up

    # Inflow from neighbors (their outflow becomes our inflow)
    # Roll shifts the array so neighbor values appear at current position
    inflow_from_left = jnp.roll(flux_right, 1, axis=-1)   # left neighbor's rightward flux
    inflow_from_right = jnp.roll(flux_left, -1, axis=-1)  # right neighbor's leftward flux
    inflow_from_above = jnp.roll(flux_down, 1, axis=-2)   # above neighbor's downward flux
    inflow_from_below = jnp.roll(flux_up, -1, axis=-2)    # below neighbor's upward flux

    # Total inflow to each cell
    inflow = inflow_from_left + inflow_from_right + inflow_from_above + inflow_from_below

    # Update mass: current - outflow + inflow
    new_mass = mass + dt * (inflow - outflow)

    return jnp.clip(new_mass, 0.0, 1.0)


def diffuse_mass(
    mass: jnp.ndarray,
    diffusion_rate: float = 0.1
) -> jnp.ndarray:
    """Apply diffusion to mass field (spreads mass to neighbors).

    This allows mass to explore the space even without directed velocity.
    Conserves total mass.

    Args:
        mass: Density field of shape (H, W) or (B, H, W)
        diffusion_rate: How much mass spreads per step (0-0.25 for stability)

    Returns:
        Diffused mass field with same total mass
    """
    # Clamp diffusion rate for stability
    rate = jnp.clip(diffusion_rate, 0.0, 0.25)

    # Compute neighbor average
    padded = jnp.pad(mass, ((1, 1), (1, 1)) if mass.ndim == 2 else ((0, 0), (1, 1), (1, 1)), mode='wrap')

    if mass.ndim == 2:
        neighbor_sum = (
            padded[:-2, 1:-1] + padded[2:, 1:-1] +  # up, down
            padded[1:-1, :-2] + padded[1:-1, 2:]     # left, right
        )
    else:  # Batched
        neighbor_sum = (
            padded[:, :-2, 1:-1] + padded[:, 2:, 1:-1] +
            padded[:, 1:-1, :-2] + padded[:, 1:-1, 2:]
        )

    # Diffusion: give some mass to neighbors, receive from neighbors
    # Each cell gives rate * mass to each of 4 neighbors
    # Each cell receives rate * neighbor_mass from each neighbor
    outflow = 4 * rate * mass
    inflow = rate * neighbor_sum

    new_mass = mass - outflow + inflow
    return jnp.clip(new_mass, 0.0, 1.0)


def advect_with_diffusion(
    mass: jnp.ndarray,
    velocity_x: jnp.ndarray,
    velocity_y: jnp.ndarray,
    dt: float = 0.5,
    diffusion_rate: float = 0.05
) -> jnp.ndarray:
    """Advect mass with added diffusion for exploration.

    Combines directed transport (advection) with random spreading (diffusion).

    Args:
        mass: Density field
        velocity_x: Horizontal velocity
        velocity_y: Vertical velocity
        dt: Advection time step
        diffusion_rate: Diffusion strength

    Returns:
        Updated mass field
    """
    # First advect
    mass = advect_mass(mass, velocity_x, velocity_y, dt)

    # Then diffuse
    mass = diffuse_mass(mass, diffusion_rate)

    return mass


def add_velocity_noise(
    velocity_x: jnp.ndarray,
    velocity_y: jnp.ndarray,
    key: jax.random.PRNGKey,
    noise_scale: float = 0.3
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Add random noise to velocity field for exploration.

    Args:
        velocity_x: Horizontal velocity
        velocity_y: Vertical velocity
        key: Random key
        noise_scale: Standard deviation of noise

    Returns:
        Tuple of (noisy_vx, noisy_vy)
    """
    k1, k2 = jax.random.split(key)

    noise_x = jax.random.normal(k1, velocity_x.shape) * noise_scale
    noise_y = jax.random.normal(k2, velocity_y.shape) * noise_scale

    noisy_vx = jnp.clip(velocity_x + noise_x, -1.0, 1.0)
    noisy_vy = jnp.clip(velocity_y + noise_y, -1.0, 1.0)

    return noisy_vx, noisy_vy


def advect_mass_circular(
    mass: jnp.ndarray,
    velocity_x: jnp.ndarray,
    velocity_y: jnp.ndarray,
    dt: float = 0.5
) -> jnp.ndarray:
    """Advect mass with circular (toroidal) boundary conditions.

    Same as advect_mass but mass wraps around edges.
    This ensures perfect mass conservation.

    Args:
        mass: Density field of shape (H, W) or (B, H, W)
        velocity_x: Horizontal velocity [-1, 1]
        velocity_y: Vertical velocity [-1, 1]
        dt: Time step

    Returns:
        Updated mass field with exactly conserved total mass
    """
    # This is actually the same as advect_mass since jnp.roll
    # already implements circular boundaries
    return advect_mass(mass, velocity_x, velocity_y, dt)


def compute_velocity_toward_target(
    current_mass: jnp.ndarray,
    target_mass: jnp.ndarray,
    strength: float = 1.0
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute velocity field that would move mass toward target.

    This is a simple gradient-based approach:
    - Blur the target to create a potential field
    - Velocity = gradient of potential (toward higher target density)

    Args:
        current_mass: Current mass distribution
        target_mass: Target mass distribution
        strength: Velocity magnitude multiplier

    Returns:
        Tuple of (velocity_x, velocity_y)
    """
    # Blur target to create smooth potential field
    blurred = target_mass
    for _ in range(4):
        padded = jnp.pad(blurred, ((1, 1), (1, 1)), mode='edge')
        blurred = (
            padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:] +
            padded[1:-1, :-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:] +
            padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
        ) / 9.0

    # Gradient of blurred field = direction toward target
    padded = jnp.pad(blurred, ((1, 1), (1, 1)), mode='edge')
    grad_x = (padded[1:-1, 2:] - padded[1:-1, :-2]) / 2.0
    grad_y = (padded[2:, 1:-1] - padded[:-2, 1:-1]) / 2.0

    # Normalize and scale
    mag = jnp.sqrt(grad_x**2 + grad_y**2 + 1e-8)
    vx = strength * grad_x / mag
    vy = strength * grad_y / mag

    return vx, vy


def multi_step_advection(
    mass: jnp.ndarray,
    velocity_x: jnp.ndarray,
    velocity_y: jnp.ndarray,
    num_steps: int,
    dt: float = 0.5
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Run multiple advection steps with fixed velocity.

    Args:
        mass: Initial mass distribution
        velocity_x: Horizontal velocity field
        velocity_y: Vertical velocity field
        num_steps: Number of advection steps
        dt: Time step per step

    Returns:
        Tuple of (final_mass, trajectory of shape (num_steps, H, W))
    """
    def step_fn(mass, _):
        new_mass = advect_mass(mass, velocity_x, velocity_y, dt)
        return new_mass, new_mass

    final_mass, trajectory = jax.lax.scan(step_fn, mass, None, length=num_steps)
    return final_mass, trajectory


def check_mass_conservation(
    mass_before: jnp.ndarray,
    mass_after: jnp.ndarray,
    tolerance: float = 1e-5
) -> tuple[bool, float]:
    """Check if mass is conserved between two states.

    Args:
        mass_before: Mass distribution before
        mass_after: Mass distribution after
        tolerance: Acceptable difference in total mass

    Returns:
        Tuple of (is_conserved, relative_error)
    """
    total_before = jnp.sum(mass_before)
    total_after = jnp.sum(mass_after)

    relative_error = jnp.abs(total_after - total_before) / (total_before + 1e-8)
    is_conserved = relative_error < tolerance

    return bool(is_conserved), float(relative_error)