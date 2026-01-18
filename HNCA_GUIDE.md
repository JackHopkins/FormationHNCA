# Hierarchical Neural Cellular Automata in JAX: Implementation Guide

Neural Cellular Automata learn local update rules that produce emergent global patterns, making them powerful for bottom-up multi-agent control and morphological self-organization. This guide synthesizes research from foundational NCA papers, JAX implementation patterns, and stability analysis to provide actionable guidance for H-NCA systems. The core insight: **pool-based training with stochastic updates creates stable attractors**, enabling patterns that self-organize, persist, and regenerate—essential properties for hierarchical multi-agent coordination.

---

## Core NCA architecture and why it works

The standard NCA architecture from Mordvintsev et al. (2020) uses **16 channels per cell**: 3 RGB + 1 alpha + 12 hidden channels for cell "memory." Each cell perceives its Moore neighborhood through fixed Sobel filters, processes this through a small MLP (~8,000 parameters), and outputs a residual state update.

**Why Sobel filters for perception?** Real biological cells rely on chemical gradients to guide development. Sobel filters estimate partial derivatives in x and y directions, forming 2D gradient vectors—a computationally cheap approximation of chemotaxis. This creates a 48-dimensional perception vector: (16 channels × 2 gradients) + 16 self-states.

**Why stochastic updates?** Traditional CA require global clock synchronization, incompatible with true self-organization. Applying a random per-cell mask (P=0.5) models asynchronous cell updates, breaks symmetry without varied initial conditions, and acts as regularization similar to dropout.

**Why pool-based training?** The critical innovation. Rather than backpropagating through thousands of timesteps, maintain a pool of ~1024 intermediate states. Sample batches from the pool, train, then inject outputs back. This creates **attractor dynamics**: the CA learns not just trajectories to targets, but how to persist at and return to targets—essential for long-term stability.

```python
# Core NCA state update equation
state_{t+1} = state_t + ds_grid × random_mask  # Residual + stochastic
alive = max_pool(alpha, 3×3) > 0.1  # Living cell masking
state = state × alive
```

---

## JAX implementation patterns and libraries

The **CAX library** (ICLR 2025) provides the most comprehensive JAX/Flax NCA implementation with 2000x speedups over traditional approaches. For lighter alternatives, **jax-nca** offers Flax-based implementations with `jax.lax.scan` optimization. Google Research's **self-organising-systems** repo contains official reference implementations including μNCA and Biomaker CA.

### Perception layer with depthwise convolutions

```python
import jax
import jax.numpy as jnp
from flax import linen as nn

def perceive(state: jnp.ndarray) -> jnp.ndarray:
    """Compute perception vector using Sobel gradients + identity."""
    sobel_x = jnp.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=jnp.float32)
    sobel_y = sobel_x.T
    channels = state.shape[-1]
    
    # Depthwise convolution: key is feature_group_count = channels
    def depthwise_conv(inputs, kernel):
        kernel_expanded = kernel[:, :, None, None]  # (3, 3, 1, 1)
        kernel_tiled = jnp.tile(kernel_expanded, (1, 1, channels, 1))
        return jax.lax.conv_general_dilated(
            inputs[None], kernel_tiled, (1, 1), 'SAME',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
            feature_group_count=channels
        )[0]
    
    grad_x = depthwise_conv(state, sobel_x)
    grad_y = depthwise_conv(state, sobel_y)
    return jnp.concatenate([state, grad_x, grad_y], axis=-1)  # 48 channels
```

### Update rule network with zero initialization

```python
class NCAUpdateRule(nn.Module):
    hidden_dim: int = 128
    num_channels: int = 16
    
    @nn.compact
    def __call__(self, perception: jnp.ndarray) -> jnp.ndarray:
        x = nn.Conv(self.hidden_dim, kernel_size=(1, 1))(perception)
        x = nn.relu(x)
        # CRITICAL: Zero initialization for "do-nothing" initial behavior
        ds = nn.Conv(self.num_channels, kernel_size=(1, 1),
                     kernel_init=nn.initializers.zeros)(x)
        return ds  # Residual update, no activation
```

### Stochastic updates with proper PRNG handling

```python
def stochastic_update(state: jnp.ndarray, ds: jnp.ndarray, 
                       key: jax.random.PRNGKey, fire_rate: float = 0.5) -> jnp.ndarray:
    """Apply stochastic cell update mask."""
    mask = jax.random.bernoulli(key, fire_rate, shape=state.shape[:2])
    mask = mask[..., None]  # Broadcast to all channels
    return state + ds * mask

def alive_masking(state: jnp.ndarray, threshold: float = 0.1) -> jnp.ndarray:
    """Zero out dead cells based on alpha channel neighborhood."""
    alpha = state[:, :, 3:4]
    alive = jax.lax.reduce_window(
        alpha, -jnp.inf, jax.lax.max, (3, 3, 1), (1, 1, 1), 'SAME'
    ) > threshold
    return state * alive.astype(jnp.float32)
```

### Efficient temporal evolution with lax.scan

```python
def multi_step_nca(params, nca_apply, state: jnp.ndarray, 
                   key: jax.random.PRNGKey, num_steps: int):
    """Memory-efficient multi-step evolution using scan."""
    keys = jax.random.split(key, num_steps)
    
    def step_fn(carry, subkey):
        state = carry
        perception = perceive(state)
        ds = nca_apply(params, perception)
        state = stochastic_update(state, ds, subkey)
        state = alive_masking(state)
        return state, state
    
    final_state, trajectory = jax.lax.scan(step_fn, state, keys)
    return final_state, trajectory
```

---

## H-NCA architecture for multi-agent systems

Hierarchical NCA (Pande & Grattarola, ALIFE 2023) models self-organizing systems at **two scales**: a parent-NCA controlling clusters of child-NCA cells. This maps directly to multi-agent coordination where higher-level agents coordinate lower-level behaviors.

### Communication layer architecture

The hierarchy communicates through three mechanisms:

| Component | Direction | Function |
|-----------|-----------|----------|
| **Sensor** | Child → Parent | Averages child cell cluster states to initialize parent-NCA state |
| **Actuator** | Parent → Child | Parent-NCA broadcasts influence via designated signal channels |
| **Multiplexer** | Bidirectional | Mixes signals by adding source channel values to destination channels |

```python
class HierarchicalNCA(nn.Module):
    """Two-scale H-NCA with sensor/actuator communication."""
    child_channels: int = 48
    parent_channels: int = 16
    cluster_size: int = 4  # 4×4 child cells per parent cell
    tau_c: int = 10  # Child steps before parent sensing
    
    @nn.compact
    def __call__(self, child_state, parent_state, key):
        key1, key2 = jax.random.split(key)
        
        # Sensor: average child clusters → parent initial state
        def sensor(child_state):
            # Pool child states into parent resolution
            return jax.lax.reduce_window(
                child_state, 0.0, jax.lax.add,
                (self.cluster_size, self.cluster_size, 1),
                (self.cluster_size, self.cluster_size, 1), 'VALID'
            ) / (self.cluster_size ** 2)
        
        # Actuator: broadcast parent signals to child signal channels
        def actuator(parent_state, child_state):
            # Upsample parent state to child resolution
            upsampled = jax.image.resize(
                parent_state, 
                (child_state.shape[0], child_state.shape[1], parent_state.shape[-1]),
                method='nearest'
            )
            # Add to child signal channels (last N channels)
            signal_channels = child_state.shape[-1] - self.parent_channels
            child_state = child_state.at[:, :, signal_channels:].add(upsampled)
            return child_state
        
        # Child NCA step
        child_state = self.child_nca(child_state, key1)
        
        # Parent NCA step (after τ_c child steps initially)
        parent_input = sensor(child_state)
        parent_state = self.parent_nca(parent_input, key2)
        
        # Actuate child from parent
        child_state = actuator(parent_state, child_state)
        
        return child_state, parent_state
```

### τ_c sensing delay handling

The **τ_c parameter** specifies how many timesteps the child-NCA evolves before the sensor averages states for the parent. This creates temporal hierarchy:

- **Initial phase**: Child runs τ_c steps alone to form basic patterns
- **Coupled phase**: Both NCAs evolve in parallel, exchanging signals each step

**Stability considerations**: Too small τ_c means noisy, unformed child patterns dominate parent sensing. Too large τ_c creates feedback instability from outdated parent information. Typical values: **τ_c = 5-15** depending on child convergence speed.

---

## Multi-agent coordination through locality

NCA principles map directly to multi-agent systems because **locality enables scalability**:

| NCA Property | Multi-Agent Benefit |
|--------------|---------------------|
| Same rule for all cells | Identical agents, simpler deployment |
| Local neighborhood only | Communication scales O(n), not O(n²) |
| Emergent global patterns | Coordination without central planner |
| Self-repair capability | Fault tolerance, graceful degradation |
| Stochastic updates | Asynchronous agent execution |

**Embodied Spiking NCA (SNCA)** extends this for voxel-based soft robots where each voxel is an NCA cell. SNNs provide spike-timing-dependent plasticity for local learning rules and native inter-module communication—achieving competitive performance with better adaptability to unforeseen environmental changes.

---

## Training stability and failure modes

Training instabilities manifest as **sudden loss spikes in later training stages**. The core causes are gradient explosion through BPTT, pattern explosion/decay without persistence training, and pool contamination.

### Common failure modes

| Failure Mode | Symptoms | Root Cause |
|--------------|----------|------------|
| **Exploding activations** | Loss → NaN, unbounded growth | No pool training; compound errors through timesteps |
| **Pattern decay** | Correct formation, later dissolution | Only trajectory learning, not attractor learning |
| **Mode collapse** | Single output regardless of input | Pool lacks diversity; overfitting to narrow states |
| **Gradient explosion** | Sudden loss spikes | BPTT through 64-96 steps without gradient clipping |
| **Catastrophic forgetting** | Loses seed→target capability | No seed reinjection in pool sampling |

### Pool-based training implementation

```python
class NCAPool:
    """Sample pool for stable NCA training."""
    def __init__(self, seed: jnp.ndarray, pool_size: int = 1024):
        self.pool = jnp.tile(seed[None], (pool_size, 1, 1, 1))
        self.seed = seed
        
    def sample(self, batch_size: int = 32, key: jax.random.PRNGKey = None):
        """Sample batch, replace highest-loss with seed."""
        idxs = jax.random.choice(key, len(self.pool), (batch_size,), replace=False)
        batch = self.pool[idxs]
        return idxs, batch
    
    def update(self, idxs, outputs, losses):
        """Inject outputs back; replace worst with seed."""
        # Sort by loss descending
        sorted_order = jnp.argsort(-losses)
        sorted_idxs = idxs[sorted_order]
        sorted_outputs = outputs[sorted_order]
        
        # Replace highest-loss sample with seed
        sorted_outputs = sorted_outputs.at[0].set(self.seed)
        self.pool = self.pool.at[sorted_idxs].set(sorted_outputs)
        
    def apply_damage(self, batch, num_damage: int = 3, key: jax.random.PRNGKey = None):
        """Damage lowest-loss samples for regeneration training."""
        # Zero out circular regions in 3 samples
        for i in range(num_damage):
            key, subkey = jax.random.split(key)
            cx, cy = jax.random.randint(subkey, (2,), 10, batch.shape[1]-10)
            radius = jax.random.randint(subkey, (), 5, 15)
            # Create circular mask and zero out
            y, x = jnp.ogrid[:batch.shape[1], :batch.shape[2]]
            mask = ((x - cx)**2 + (y - cy)**2) <= radius**2
            batch = batch.at[i, mask].set(0.0)
        return batch
```

### Gradient stabilization

```python
import optax

def create_optimizer(learning_rate: float = 2e-3):
    """Optimizer with gradient normalization for NCA stability."""
    return optax.chain(
        optax.clip_by_global_norm(1.0),  # Gradient clipping
        optax.adam(learning_rate)
    )

# Per-variable L2 normalization (Growing NCA approach)
def normalize_gradients(grads):
    """Normalize gradients per-variable to unit norm."""
    def norm_grad(g):
        return g / (jnp.linalg.norm(g) + 1e-8)
    return jax.tree_map(norm_grad, grads)
```

---

## Hyperparameter quick reference

### Core architecture parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Total channels | 16 | 4 RGBA + 12 hidden (minimum for complex coordination) |
| Hidden layer neurons | 128 | Balance expressivity vs. overfitting |
| Perception kernel | 3×3 Sobel | Gradient estimation; biologically plausible |
| Total parameters | ~8,000 | Surprisingly small; locality constraint |
| Fire rate (δ) | 0.5 | Standard; vary [0.0, 0.75] for robustness training |
| Alpha threshold | 0.1 | Living/dead cell demarcation |

### Training parameters

| Parameter | Range | Default | Notes |
|-----------|-------|---------|-------|
| Learning rate | 1e-4 to 2e-3 | 2e-3 with decay | Decay to 1e-4 over training |
| Steps per sample | 32-96 | [64, 96] random | Random sampling prevents fixed-time overfitting |
| Batch size | 8-64 | 32 | Larger batches more stable |
| Pool size | 256-1024 | 1024 | Larger pools = more diversity |
| Training iterations | 2,000-10,000 | 4,000-8,000 | Monitor convergence |
| Damage samples | 3 per batch | 3 lowest-loss | For regeneration capability |

### Loss function selection

| Task | Loss Function | Implementation |
|------|---------------|----------------|
| Morphogenesis | MSE (L2) | `jnp.mean((output[:,:,:4] - target)**2)` |
| Texture synthesis | VGG Gram matrix | Extract features from VGG block[1-5]_conv1, L2 on Gram matrices |
| Distribution matching | Sliced Wasserstein | Better for misaligned images, stationary statistics |
| State regularization | Overflow loss | `jnp.mean(jax.nn.relu(output - 1) + jax.nn.relu(-output))` |

---

## Do's and Don'ts backed by research

### ✅ Do

| Practice | Why |
|----------|-----|
| **Use circular/periodic padding** | Eliminates edge artifacts; cells see consistent physics everywhere |
| **Initialize final layer weights to zero** | Ensures "do-nothing" initial behavior; prevents explosion |
| **Always use pool-based training** | Creates attractors, not just trajectories; enables long-term stability |
| **Replace highest-loss sample with seed each batch** | Prevents catastrophic forgetting of seed→target capability |
| **Apply damage augmentation during training** | Builds regeneration; expands basin of attraction |
| **Use stochastic updates (fire_rate=0.5)** | Removes clock dependency; improves robustness |
| **Clip gradients or use per-variable normalization** | Prevents training instabilities from BPTT |
| **Sample random step counts [64, 96]** | Ensures stability across iteration counts |
| **Use depthwise convolutions for perception** | Efficient; maintains interpretability |
| **Apply alive masking after every update** | Prevents noise accumulation in dead cells |

### ❌ Don't

| Anti-pattern | Consequence |
|--------------|-------------|
| **Zero padding instead of circular** | Edge artifacts; different physics at boundaries |
| **ReLU on final output layer** | Prevents negative updates; breaks residual learning |
| **Training without pool sampling** | Pattern either dies or explodes after training steps |
| **Fixed iteration count only** | Pattern works at step 96, breaks at 150 |
| **Fully learnable perception kernels** | Reduces interpretability; harder analysis |
| **More than 9 kernels for 3×3** | Linearly dependent; no information gain |
| **Skip seed reinjection** | Pool fills with degenerate states |
| **High learning rate without gradient normalization** | Sudden loss spikes; training instability |
| **Synchronous-only updates** | Clock dependency; symmetry issues |
| **RGB channels non-zero in seed** | Seed invisible on white background |

---

## Complete training loop example

```python
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax

class NCA(nn.Module):
    num_channels: int = 16
    hidden_dim: int = 128
    fire_rate: float = 0.5
    
    @nn.compact
    def __call__(self, state, key):
        # Perception
        perception = perceive(state)
        
        # Update rule
        x = nn.Conv(self.hidden_dim, (1, 1))(perception)
        x = nn.relu(x)
        ds = nn.Conv(self.num_channels, (1, 1), 
                     kernel_init=nn.initializers.zeros)(x)
        
        # Stochastic update
        mask = jax.random.bernoulli(key, self.fire_rate, state.shape[:2])
        state = state + ds * mask[..., None]
        
        # Alive masking
        state = alive_masking(state)
        return state

def create_seed(h: int, w: int, channels: int = 16) -> jnp.ndarray:
    """Create centered seed with alpha and hidden channels set."""
    seed = jnp.zeros((h, w, channels))
    seed = seed.at[h//2, w//2, 3:].set(1.0)  # Alpha + hidden = 1
    return seed

@jax.jit
def train_step(state, batch, target, key, num_steps):
    """Single training step with pool sampling."""
    
    def loss_fn(params):
        keys = jax.random.split(key, num_steps)
        
        def step(carry, subkey):
            return nca.apply({'params': params}, carry, subkey), None
        
        final, _ = jax.lax.scan(step, batch, keys)
        loss = jnp.mean((final[:, :, :, :4] - target) ** 2)
        return loss, final
    
    (loss, final), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    grads = normalize_gradients(grads)
    state = state.apply_gradients(grads=grads)
    return state, loss, final

# Main training loop
def train_nca(target, num_epochs=8000):
    key = jax.random.PRNGKey(42)
    h, w = target.shape[:2]
    
    # Initialize
    seed = create_seed(h + 16, w + 16)  # Padding for growth
    nca = NCA()
    params = nca.init(jax.random.PRNGKey(0), seed, jax.random.PRNGKey(1))
    
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(2e-3)
    )
    state = train_state.TrainState.create(
        apply_fn=nca.apply, params=params['params'], tx=optimizer
    )
    
    pool = NCAPool(seed, pool_size=1024)
    
    for epoch in range(num_epochs):
        key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
        
        # Sample from pool
        idxs, batch = pool.sample(32, subkey1)
        
        # Apply damage to lowest-loss samples (for regeneration)
        if epoch > 1000:
            batch = pool.apply_damage(batch, num_damage=3, key=subkey2)
        
        # Random step count [64, 96]
        num_steps = jax.random.randint(subkey3, (), 64, 97)
        
        # Train
        state, loss, outputs = train_step(state, batch, target, subkey3, num_steps)
        
        # Update pool
        losses = jnp.mean((outputs[:, :, :, :4] - target) ** 2, axis=(1, 2, 3))
        pool.update(idxs, outputs, losses)
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch}: loss = {loss:.6f}")
    
    return state.params
```

---

## Edge cases and numerical stability

### Circular padding implementation

```python
def circular_pad(x, pad=1):
    """Circular (wrap-around) padding for toroidal topology."""
    return jnp.pad(x, ((pad, pad), (pad, pad), (0, 0)), mode='wrap')

# In Flax Conv, use padding='CIRCULAR' or manual padding
class CircularConv(nn.Module):
    features: int
    kernel_size: tuple = (3, 3)
    
    @nn.compact
    def __call__(self, x):
        x = circular_pad(x, pad=1)
        return nn.Conv(self.features, self.kernel_size, padding='VALID')(x)
```

### Preventing overflow in long evolutions

```python
def overflow_loss(state, min_val=0.0, max_val=1.0):
    """Auxiliary loss to keep states bounded."""
    overflow = jax.nn.relu(state - max_val) + jax.nn.relu(min_val - state)
    return jnp.mean(overflow)

def soft_clamp(x, min_val=-3.0, max_val=3.0):
    """Soft clamping using tanh for gradients."""
    scale = (max_val - min_val) / 2
    offset = (max_val + min_val) / 2
    return scale * jnp.tanh((x - offset) / scale) + offset
```

### Seed initialization strategies

```python
def single_seed(h, w, channels=16):
    """Standard single center seed."""
    seed = jnp.zeros((h, w, channels))
    return seed.at[h//2, w//2, 3:].set(1.0)

def multi_seed_isotropic(h, w, channels=16, separation=20):
    """Two seeds for isotropic NCA symmetry breaking."""
    seed = jnp.zeros((h, w, channels))
    seed = seed.at[h//2, w//2 - separation//2, 3:].set(1.0)
    seed = seed.at[h//2, w//2 + separation//2, 3:].set([0.5] * (channels-3))
    return seed

def genome_seed(h, w, channels=16, genome_bits=4, target_id=0):
    """Seed with genome encoding for multi-target NCA."""
    seed = jnp.zeros((h, w, channels))
    # Encode target ID in first genome_bits hidden channels
    genome = jnp.array([(target_id >> i) & 1 for i in range(genome_bits)])
    seed = seed.at[h//2, w//2, 3:3+genome_bits].set(genome)
    seed = seed.at[h//2, w//2, 3+genome_bits:].set(1.0)
    return seed
```

---

## Key papers and resources

| Resource | Focus | URL/Reference |
|----------|-------|---------------|
| **Growing Neural Cellular Automata** | Core architecture, pool training | Distill.pub, Mordvintsev et al. 2020 |
| **Self-Organising Textures** | Gram matrix loss, VGG features | Distill.pub, Niklasson et al. 2021 |
| **Goal-Guided NCA** | External control, goal encoding | Sudhakaran et al. 2022 |
| **Hierarchical NCA** | Parent-child hierarchy, τ_c | Pande & Grattarola, ALIFE 2023 |
| **CAX Library** | JAX implementation, 2000x speedup | github.com/maxencefaldor/cax |
| **jax-nca** | Flax implementation | github.com/shyamsn97/jax-nca |
| **google-research/self-organising-systems** | Official reference implementations | GitHub |

---

## Conclusion

Implementing H-NCA for multi-agent control requires balancing **architectural simplicity with training sophistication**. The ~8,000 parameter network learns surprisingly complex behaviors because locality constraints force efficient information encoding. Pool-based training with stochastic updates creates robust attractors that enable self-repair and persistence—properties that transfer directly to fault-tolerant multi-agent coordination.

Three critical insights emerge: First, **zero-initialize the final layer** to prevent early explosion. Second, **pool training is non-negotiable** for stability beyond training horizons. Third, **circular padding and stochastic updates** aren't optional embellishments but fundamental to correct CA physics. The H-NCA extension adds sensor/actuator communication layers that map naturally to hierarchical multi-agent architectures, with τ_c controlling the temporal coupling between scales.

For soft robot and morphogenesis applications, consider SNCA variants that leverage spike-timing-dependent plasticity for neuromorphic hardware compatibility and better environmental adaptability.