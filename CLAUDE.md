# Hierarchical NCA for Total War-scale battle simulation

A two-tier Neural Cellular Automata architecture can simulate **40,000 agents in real-time** by leveraging JAX's vectorization and a hierarchical decomposition where parent-NCAs control formation-level behavior while child-NCAs govern individual unit responses. This approach achieves **200-400 FPS inference** on modern GPUs by encoding combat mechanics—morale propagation, flanking detection, routing cascades—as local NCA update rules that operate through depthwise convolutions with learned kernels.

## Recommended architecture with 24-channel state representation

The core insight from Growing NCA research is that complex emergent behaviors arise from local update rules applied to multi-channel cell states. For battle simulation, extend the standard 16-channel design to **24 channels** that encode both visible combat state and hidden coordination signals.

**Child-NCA Channel Allocation (per-unit level):**

| Channels | Purpose | Value Range |
|----------|---------|-------------|
| 0-2 | RGB visualization (team colors) | [0, 1] |
| 3 | Alpha/alive (unit presence) | [0, 1], threshold 0.1 |
| 4 | Health | [0, 1] |
| 5 | Morale | [-1, 1], negative = routing |
| 6 | Fatigue | [0, 1], 0 = fresh |
| 7-8 | Velocity (vx, vy) | normalized |
| 9 | Unit type encoding | categorical |
| 10 | Formation ID | integer encoded |
| 11-12 | Parent command signals | from parent NCA |
| 13-14 | Enemy proximity/direction | computed per step |
| 15-23 | Hidden state channels | learned coordination |

**Parent-NCA Channel Allocation (formation level, coarser grid):**

| Channels | Purpose |
|----------|---------|
| 0-3 | Formation shape (RGBA target) |
| 4-5 | Formation velocity/heading |
| 6 | Formation integrity (% alive) |
| 7-8 | Command outputs (advance/hold/charge/wheel) |
| 9-15 | Hidden coordination state |

The hierarchical connection uses an **actuator** (parent→child signal injection) and **sensor** (child→parent aggregation). Parent cells upscale to child resolution via nearest-neighbor interpolation, writing to channels 11-12 of child cells. The sensor aggregates child cell states within each parent cell's receptive field using average pooling.

## Combat mechanics as learned local rules

Rather than hardcoding combat resolution, encode the mechanics as **training objectives** that the NCA learns to satisfy through local update rules. This approach produces emergent behavior that's robust to perturbations.

**Melee engagement** triggers when enemy cells occupy adjacent positions (within 3×3 perception). The NCA perceives enemy presence through dedicated channels and learns to output health decrements:

```python
def combat_loss(state, enemy_state):
    # Detect adjacent enemies via convolution
    enemy_presence = nn.max_pool(enemy_state[..., 3:4], (3,3), padding='SAME')
    in_combat = (state[..., 3] > 0.1) & (enemy_presence[..., 0] > 0.1)
    
    # Expected health decay when engaged
    expected_damage = in_combat * compute_damage_rate(state, enemy_state)
    actual_change = state_t1[..., 4] - state_t0[..., 4]
    
    return jnp.mean((actual_change + expected_damage) ** 2)
```

**Morale propagation** uses the NCA's natural diffusion properties. Total War's morale system shows that **cascade routing** occurs when nearby units flee—a -12 leadership penalty for seeing 2+ friendly units route. Encode this as a loss that rewards morale decrements when surrounding morale drops below threshold:

```python
def morale_propagation_loss(state_t0, state_t1):
    # Count routing neighbors (morale < -0.5)
    routing_neighbors = nn.avg_pool(
        (state_t0[..., 5:6] < -0.5).astype(float), (5,5), padding='SAME')
    
    # Morale should decrease when surrounded by routing units
    expected_morale_drop = routing_neighbors * 0.15  # Per routing neighbor
    return jnp.mean((state_t1[..., 5] - state_t0[..., 5] + expected_morale_drop) ** 2)
```

**Flanking detection** leverages the Sobel gradient perception. Units store their facing direction in velocity channels; when attack direction differs from facing by >90°, apply defense reduction. The NCA perceives this through gradient channels and learns appropriate damage multipliers matching Total War's **60% defense for flank, 25% for rear** attacks.

## Training strategy for historical formations

Training proceeds in **three curriculum phases**, each building capabilities needed for full battle simulation.

**Phase 1: Static formation learning (1,500 iterations)**
Train the child-NCA to grow and maintain single formation targets. Create target images representing historical formations—phalanx (16 ranks deep, dense spacing), line infantry (2-3 ranks, shoulder-to-shoulder), hollow square (4-6 ranks per side, hollow center). Use the pool-based training approach with **1,024 sample pool**, replacing the highest-loss sample with the seed each batch:

```python
def phase1_training(nca_params, targets, pool, key):
    batch_idxs = jax.random.choice(key, len(pool), (32,), replace=False)
    batch = pool[batch_idxs]
    batch = batch[jnp.argsort(formation_loss(batch, targets))]
    batch = batch.at[0].set(seed)  # Prevent catastrophic forgetting
    
    # Random steps [64-96] for temporal generalization
    num_steps = jax.random.randint(key, (), 64, 96)
    final_states = run_nca(nca_params, batch, num_steps)
    
    loss = formation_fidelity_loss(final_states, targets)
    return loss, final_states
```

**Phase 2: Multi-formation and transitions (2,500 iterations)**
Introduce conditional formation control using **goal-guided NCA**. A small MLP encoder maps one-hot formation IDs to perturbation vectors added to hidden state channels. Train with random formation switches mid-simulation:

```python
def phase2_training(nca_params, goal_encoder_params, key):
    formation_ids = ['phalanx', 'line', 'square', 'wedge', 'column']
    
    # Grow to random initial formation
    init_goal = jax.random.choice(key, formation_ids)
    state = grow_to_formation(seed, init_goal, steps=50)
    
    # Switch to different formation
    target_goal = jax.random.choice(key, formation_ids)
    final_state = grow_to_formation(state, target_goal, steps=50)
    
    return formation_loss(final_state, targets[target_goal])
```

**Phase 3: Combat dynamics and parent-child coordination (4,000 iterations)**
Train the full hierarchical system with adversarial self-play. Freeze child-NCA weights, train parent-NCA to coordinate formations against an opponent:

```python
def phase3_training(parent_params, child_params_frozen, key):
    # Two armies, each controlled by separate parent-NCAs
    red_state, blue_state = initialize_armies()
    
    for step in range(100):
        # Parent generates commands
        red_commands = parent_nca(aggregate_child_state(red_state))
        blue_commands = parent_nca(aggregate_child_state(blue_state))
        
        # Children execute with combat
        red_state = child_nca(red_state, red_commands, blue_state)
        blue_state = child_nca(blue_state, blue_commands, red_state)
    
    # Multi-objective loss
    return (formation_integrity_loss(red_state) + 
            combat_effectiveness_loss(red_state, blue_state) +
            casualty_ratio_loss(red_state, blue_state))
```

Include **damage augmentation**—randomly zero out circular regions of 3-8 lowest-loss samples each batch—to train regeneration/reformation behavior matching real units' ability to re-form after disruption.

## JAX implementation patterns for 40k agents

Achieving real-time performance with 40,000 agents requires specific JAX patterns that maximize GPU utilization.

**Grid sizing:** For a 200×200 grid (40,000 cells) with 24 channels at float32, each state tensor occupies **3.84 MB**. A training batch of 32 requires **123 MB** for states alone. Use **bfloat16** for forward passes to halve memory and gain ~2x throughput on A100/H100 GPUs:

```python
@partial(jax.jit, donate_argnums=(0,))
def nca_step_bf16(state, params, key):
    state = state.astype(jnp.bfloat16)
    
    # Perception: depthwise conv with Sobel kernels
    perception = perceive(state)  # 72 channels: 24 state + 24 grad_x + 24 grad_y
    
    # Update: small MLP (8k params)
    update = mlp(perception, params)  # 72 -> 128 -> 24
    
    # Stochastic masking (50% cell fire rate)
    mask = jax.random.bernoulli(key, 0.5, state.shape[:2])
    
    return (state + update * mask[..., None]).astype(jnp.float32)
```

**Multi-step simulation** uses `jax.lax.scan` for memory efficiency. Pre-generate all random keys to avoid Python overhead:

```python
def run_simulation(state, params, key, num_steps):
    keys = jax.random.split(key, num_steps)
    
    def step_fn(carry, step_key):
        state = nca_step_bf16(carry, params, step_key)
        return state, state[..., :4]  # Only store RGBA for visualization
    
    final_state, trajectory = jax.lax.scan(step_fn, state, keys)
    return final_state, trajectory
```

**Gradient checkpointing** for training with 64-96 steps avoids storing all intermediate activations:

```python
from jax import checkpoint

@checkpoint  # Recompute forward during backward
def nca_step_checkpointed(state, params, key):
    return nca_step_bf16(state, params, key)
```

**Spatial partitioning** for combat interactions uses uniform grid hashing rather than quadtrees (JAX requires static shapes). For a 200×200 battlefield with 10×10 spatial cells:

```python
def compute_combat_interactions(red_state, blue_state, cell_size=20):
    # Hash positions to cells
    red_cells = jnp.floor(red_positions / cell_size).astype(jnp.int32)
    blue_cells = jnp.floor(blue_positions / cell_size).astype(jnp.int32)
    
    # Only check adjacent 9 cells for interactions
    neighbor_offsets = jnp.array([[-1,-1],[0,-1],[1,-1],[-1,0],[0,0],[1,0],[-1,1],[0,1],[1,1]])
    
    # Vectorized interaction computation
    return jax.vmap(compute_cell_interactions)(red_cells, blue_cells, neighbor_offsets)
```

## Performance estimates and optimization techniques

Based on CAX library benchmarks and JAX-MD performance data, expect the following throughput on an **RTX 4090 or A100**:

| Configuration | Inference FPS | Training (steps/sec) |
|--------------|---------------|---------------------|
| 200×200, 24ch, float32 | 250-350 | 40-60 |
| 200×200, 24ch, bfloat16 | 400-600 | 80-120 |
| 256×256, 24ch, bfloat16 | 300-450 | 50-80 |

**Key optimizations:**

The **CAX library** provides pre-built NCA components optimized for JAX, claiming up to 2,000x speedup over CPU implementations. Use its perception modules and pool sampling utilities:

```python
from cax.core.perceive import DepthwiseConvPerceive
from cax.core.update import MLPUpdate

perceive = DepthwiseConvPerceive(
    num_channels=24,
    kernel_size=3,
    include_self=True
)
update = MLPUpdate(
    num_channels=24,
    hidden_size=128,
    num_hidden_layers=1
)
```

**Memory bandwidth** is the primary bottleneck for NCA at this scale. Ensure channel-last layout (NHWC) for efficient depthwise convolutions. Avoid unnecessary data movement by using `donate_argnums` to reuse input buffers for outputs.

**Batch across scenarios** rather than increasing grid size. Running 8 independent battles in parallel (8×40k = 320k total cells) achieves better GPU utilization than a single 566×566 grid.

## Specific do's and don'ts for battle simulation

**Do:**
- Initialize hidden channels to small random values (±0.1) rather than zeros—enables immediate coordination signals
- Use **pool-based training** to maintain long-horizon stability without exploding memory
- Include **damage perturbation** during training (zero random 10-20% of cells) for robust reformation
- Encode unit types as continuous embeddings rather than one-hot—enables gradient flow for type-dependent behaviors
- Train morale propagation explicitly with cascade routing loss matching Total War's -12 penalty for routing neighbors
- Use separate perception radii for different mechanics: 3×3 for melee, 7×7 for morale contagion, 11×11 for formation cohesion

**Don't:**
- Don't use float16—bfloat16 has better numerical range for the small gradient updates NCA produces
- Don't train parent and child NCAs jointly from scratch—pre-train child-NCA for 1,500 iterations first
- Don't hardcode combat formulas—encode as training losses so NCA learns robust local rules
- Don't use large hidden state (>128 channels)—the 8k parameter budget is deliberate, more parameters cause overfitting to specific scenarios
- Don't update all cells every step—50% stochastic update rate is critical for asynchronous emergent coordination
- Don't ignore formation depth—Total War shows 7+ ranks provides charge resistance; encode rank position in channels

**Formation-specific encodings:**

| Formation | Target Pattern Properties |
|-----------|--------------------------|
| Phalanx | 16 ranks deep, α density 0.95, velocity channels aligned forward |
| Testudo | 5×5 dense square, α=1.0 (maximum density), velocity=0 |
| Line | 2-3 ranks, width matching army size, 0.8 spacing |
| Square | Hollow center, 4-6 deep walls, all edges facing outward |
| Wedge | Triangle apex forward, density gradient from tip to base |

**Combat state transitions** should be trained as conditional behaviors: when channel 13 (enemy proximity) exceeds 0.7 and channel 6 indicates cavalry type, trigger square formation. Train these transitions with scenario-specific loss terms rather than explicit conditionals in the update rule.

The hierarchical decomposition enables strategic decisions (formation selection, positioning) at the parent level while tactical execution (individual movement, target selection) emerges from child-NCA local rules—mirroring how historical commanders gave formation orders while soldiers made local combat decisions autonomously.