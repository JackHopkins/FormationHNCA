# JAX Performance Optimization: The Complete Cheatsheet

Maximizing throughput in JAX requires understanding its tracing-based compilation model and XLA optimization pipeline. This cheatsheet distills **40+ patterns** with concrete examples—what to do, what to avoid, and why each decision matters for performance. JAX's speed comes not from magic but from enabling XLA to see your entire computation graph and optimize it holistically.

---

## JIT compilation: The foundation of JAX performance

The single most impactful optimization is proper use of `jax.jit`. Without JIT, JAX dispatches each operation as a separate GPU kernel. With JIT, XLA compiles your entire computation into **fused, optimized kernels** that minimize memory traffic and maximize hardware utilization.

### JIT at the outermost possible scope

XLA can only optimize what it can see. Give it the largest possible computation chunk.

**✅ DO:**
```python
@jax.jit
def train_step(params, batch, optimizer_state):
    def loss_fn(p):
        predictions = model(p, batch['x'])
        return jnp.mean((predictions - batch['y'])**2)
    
    loss, grads = jax.value_and_grad(loss_fn)(params)
    params = update_params(params, grads, optimizer_state)
    return params, loss
```

**❌ DO NOT:**
```python
# Separate JIT calls prevent cross-function optimization
loss_fn_jit = jax.jit(loss_fn)
grad_fn_jit = jax.jit(jax.grad(loss_fn))
update_jit = jax.jit(update_params)

def train_step(params, batch, optimizer_state):
    loss = loss_fn_jit(params, batch)  # Barrier
    grads = grad_fn_jit(params, batch)  # Barrier  
    return update_jit(params, grads, optimizer_state), loss
```

**Why it matters:** Each JIT boundary is a compilation barrier. XLA cannot fuse operations across these boundaries—intermediate results must be materialized to memory. A single large JIT allows kernel fusion, elimination of intermediate buffers, and global scheduling optimizations. The performance difference can be **2-10x** for complex models.

### Avoiding recompilation with static arguments

JAX caches compiled functions based on `(function_hash, static_arg_values, array_shapes, dtypes)`. Changing any of these triggers recompilation—which can take seconds to minutes.

**✅ DO:**
```python
from functools import partial

@partial(jax.jit, static_argnames=['num_layers', 'activation'])
def forward_pass(params, x, num_layers, activation='relu'):
    for i in range(num_layers):  # Unrolled at compile time
        x = linear(params[i], x)
        x = ACTIVATIONS[activation](x)
    return x

# Good: Limited set of static values (e.g., num_layers ∈ {2, 4, 6})
result = forward_pass(params, x, num_layers=4, activation='relu')
```

**❌ DO NOT:**
```python
@partial(jax.jit, static_argnames=['learning_rate'])
def update(params, grads, learning_rate):
    return jax.tree.map(lambda p, g: p - learning_rate * g, params, grads)

# BAD: learning_rate changes every call → recompilation every call!
for step in range(10000):
    lr = 0.1 * (0.99 ** step)  # Unique value each time
    params = update(params, grads, learning_rate=lr)  # 10000 compilations!
```

**Why it matters:** Static arguments should only be used for values with a **small, finite set** of possibilities (like model architecture choices), never for continuously-varying values like learning rates or batch sizes. TPU compilation can take minutes for large models—recompiling on every step is catastrophic.

**Detection tip:** Enable `jax.config.update('jax_log_compiles', True)` to see every recompilation.

### Warmup before benchmarking or production

JIT compilation is lazy—it happens on first call. The first invocation includes tracing + XLA compilation, which is **100-1000x slower** than execution.

**✅ DO:**
```python
@jax.jit
def inference(params, x):
    return model(params, x)

# Warmup with representative input
dummy_x = jnp.zeros((batch_size, input_dim))
_ = inference(params, dummy_x)
jax.block_until_ready(_)  # Ensure compilation completes

# Now ready for production with consistent latency
for batch in data_stream:
    result = inference(params, batch)
```

**❌ DO NOT:**
```python
# First iteration includes compilation time
for i, batch in enumerate(data_stream):
    start = time.time()
    result = inference(params, batch)  # First call: 10 seconds. Rest: 0.01 seconds
    print(f"Iteration {i}: {time.time() - start:.2f}s")  # Misleading!
```

---

## Control flow: The tracer-friendly way

JAX traces functions by replacing arrays with abstract "tracers" that track shapes and dtypes but carry no values. Python control flow operates at trace time, not runtime—causing either errors or unintended behavior.

### Use JAX control flow primitives, not Python

**✅ DO:**
```python
from jax import lax

@jax.jit
def safe_divide(x, y):
    return lax.cond(
        y != 0,
        lambda operand: operand[0] / operand[1],
        lambda operand: 0.0,
        (x, y)
    )

@jax.jit
def iterative_computation(init_val, num_iters):
    def body_fn(i, val):
        return val * 1.01 + jnp.sin(val)
    return lax.fori_loop(0, num_iters, body_fn, init_val)
```

**❌ DO NOT:**
```python
@jax.jit
def bad_divide(x, y):
    if y != 0:  # ERROR: TracerBoolConversionError
        return x / y
    else:
        return 0.0

@jax.jit
def bad_loop(arr, n):
    total = 0.0
    for i in range(n):  # If n is traced: unrolls at trace time or fails
        total += arr[i]
    return total
```

**Why it matters:** During tracing, `y` is a tracer without a concrete value—Python's `if` statement can't evaluate it. Python `for` loops with traced bounds either fail or unroll completely, generating massive XLA programs (I've seen 100,000+ line jaxpr outputs that take 20 minutes to compile).

**Control flow primitive reference:**

| Construct | JIT Compatible | Differentiable | Best For |
|-----------|---------------|----------------|----------|
| `lax.cond` | ✓ | Both branches | Value-dependent branching |
| `lax.switch` | ✓ | All branches | Multi-way branching |
| `lax.fori_loop` | ✓ | Static bounds only | Fixed iteration count |
| `lax.while_loop` | ✓ | Forward only | Dynamic termination |
| `lax.scan` | ✓ | ✓ | Sequential with outputs |

### Prefer scan over fori_loop for gradient efficiency

**✅ DO:**
```python
@jax.jit
def rnn_forward(params, inputs):
    def step(hidden, x):
        new_hidden = jnp.tanh(hidden @ params['Wh'] + x @ params['Wx'])
        return new_hidden, new_hidden  # (carry, output)
    
    final_hidden, all_hiddens = lax.scan(step, jnp.zeros(hidden_size), inputs)
    return all_hiddens
```

**❌ DO NOT:**
```python
@jax.jit
def rnn_forward_bad(params, inputs):
    def body(i, carry):
        hidden, outputs = carry
        new_hidden = jnp.tanh(hidden @ params['Wh'] + inputs[i] @ params['Wx'])
        return (new_hidden, outputs.at[i].set(new_hidden))
    
    init = (jnp.zeros(hidden_size), jnp.zeros((len(inputs), hidden_size)))
    _, all_hiddens = lax.fori_loop(0, len(inputs), body, init)
    return all_hiddens
```

**Why it matters:** For autodiff, `scan` saves only per-iteration slices of the carry, while `fori_loop` snapshots the **entire** carry at each step. For RNNs and transformers with long sequences, this can be the difference between fitting in memory and OOM.

---

## Vectorization with vmap: Batch everything

`jax.vmap` transforms a function operating on single examples into one operating on batches—with **zero Python overhead**. It's typically **10-100x faster** than Python loops.

### Always vmap instead of Python loops

**✅ DO:**
```python
def predict_single(params, x):
    for W, b in params:
        x = jnp.tanh(jnp.dot(x, W) + b)
    return x

# Vectorize over batch dimension
batched_predict = jax.jit(jax.vmap(predict_single, in_axes=(None, 0)))
outputs = batched_predict(params, batch_inputs)  # Single fused kernel
```

**❌ DO NOT:**
```python
def predict_batch_manual(params, batch_inputs):
    outputs = []
    for x in batch_inputs:  # Python loop → sequential execution
        out = predict_single(params, x)
        outputs.append(out)
    return jnp.stack(outputs)
```

### Correctly specify in_axes for mixed batched/unbatched arguments

**✅ DO:**
```python
# Batch over inputs (axis 0), but NOT over weights or bias
batched_linear = jax.vmap(linear_layer, in_axes=(0, None, None))
#                                                 ↑   ↑      ↑
#                                                 x   W      b
result = batched_linear(batched_x, weights, bias)  # W, b broadcast
```

**❌ DO NOT:**
```python
def bad_batched_linear(batched_x, weights, bias):
    # DON'T manually replicate weights—wastes memory!
    replicated_weights = jnp.stack([weights] * len(batched_x))
    replicated_bias = jnp.stack([bias] * len(batched_x))
    return jnp.einsum('bi,bij->bj', batched_x, replicated_weights) + replicated_bias
```

### Per-example gradients: The canonical pattern

**✅ DO:**
```python
def loss_single(params, x, y):
    pred = model(params, x)
    return jnp.sum((pred - y) ** 2)

# Per-example gradients (essential for differential privacy, Fisher info)
per_example_grads = jax.jit(jax.vmap(jax.grad(loss_single), in_axes=(None, 0, 0)))
grads = per_example_grads(params, x_batch, y_batch)  # Shape: (batch_size, *param_shapes)
```

---

## Multi-device parallelism: pmap and shard_map

For multi-GPU/TPU training, JAX provides data and model parallelism primitives. **Note:** `pmap` is the older API; `shard_map` is recommended for new code.

### Data parallelism with pmap

**✅ DO:**
```python
from jax import pmap
import jax.numpy as jnp

@pmap
def parallel_train_step(params, batch):
    grads = jax.grad(loss_fn)(params, batch)
    # Average gradients across all devices
    grads = jax.lax.pmean(grads, axis_name='i')
    return jax.tree.map(lambda p, g: p - 0.01 * g, params, grads)

parallel_train_step = pmap(parallel_train_step, axis_name='i')

# Replicate params and shard data
n_devices = jax.device_count()
replicated_params = jax.tree.map(lambda x: jnp.stack([x] * n_devices), params)
sharded_batch = batch.reshape(n_devices, per_device_batch_size, -1)

new_params = parallel_train_step(replicated_params, sharded_batch)
```

**❌ DO NOT:**
```python
@pmap
def bad_parallel_step(params, batch):
    grads = jax.grad(loss_fn)(params, batch)
    # Forgetting pmean → each device diverges independently!
    return jax.tree.map(lambda p, g: p - 0.01 * g, params, grads)
```

**Why it matters:** Without `pmean` (or `psum`), each device computes gradients only from its local data shard. Model replicas diverge immediately, destroying training.

### Modern approach: shard_map with explicit sharding

The JAX team recommends `shard_map` over `pmap` for new code—it's more flexible and composes better with `jit`.

**✅ DO:**
```python
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental.shard_map import shard_map
from functools import partial

mesh = Mesh(jax.devices(), ('data',))

@jax.jit
@partial(shard_map, mesh=mesh, in_specs=P('data'), out_specs=P('data'))
def parallel_fn(x):
    # x is local shard (shape = global_shape / num_devices)
    result = expensive_computation(x)
    return result

# Visualize sharding to verify
jax.debug.visualize_array_sharding(sharded_array)
```

### Combine vmap inside pmap for maximum utilization

**✅ DO:**
```python
@pmap
def device_step(params, device_batch):
    # pmap: inter-device parallelism
    # vmap: intra-device vectorization (uses SIMD/tensor cores)
    per_example_loss = jax.vmap(loss_fn, in_axes=(None, 0))(params, device_batch)
    return jnp.mean(per_example_loss)
```

---

## Memory management: Maximize hardware utilization

JAX's memory model differs from NumPy. Understanding it prevents OOM errors and maximizes throughput.

### Buffer donation for in-place-style updates

When an input buffer won't be reused, tell XLA it can recycle the memory for outputs.

**✅ DO:**
```python
@jax.jit(donate_argnums=(0, 1))  # Donate params and opt_state
def train_step(params, opt_state, batch):
    grads = compute_gradients(params, batch)
    new_params = apply_updates(params, grads)
    new_opt_state = update_optimizer(opt_state, grads)
    return new_params, new_opt_state

# After this call, old params/opt_state buffers are INVALID
params, opt_state = train_step(params, opt_state, batch)
```

**❌ DO NOT:**
```python
@jax.jit(donate_argnums=(0,))
def update_and_log(params, grads):
    new_params = apply_updates(params, grads)
    return new_params

new_params = update_and_log(params, grads)
print(params)  # ERROR: Buffer was donated and invalidated!
```

**Why it matters:** Without donation, XLA allocates new memory for outputs while keeping inputs alive—doubling peak memory. With donation, XLA can reuse input buffers, reducing peak memory by up to **50%** for update operations.

### Gradient checkpointing for deep networks

Trade compute for memory by recomputing activations during backprop instead of storing them.

**✅ DO:**
```python
from functools import partial

# Checkpoint transformer blocks—recompute activations during backward
@partial(jax.checkpoint, 
         policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable)
def transformer_block(x, params):
    attn_out = attention(x, params['attn'])
    return mlp(attn_out, params['mlp'])

# For sequential layers, combine with scan
def forward(params_list, x):
    @jax.checkpoint
    def block(x, params):
        return layer(x, params), None
    final, _ = lax.scan(block, x, params_list)
    return final
```

**Why it matters:** Transformers store **O(layers × batch × seq × hidden)** activations for backprop. Checkpointing reduces this to **O(layers)** by recomputing within each block. Memory savings of **90%+** for deep models, at cost of ~20-30% more compute.

**Checkpoint policy options:**

| Policy | Saves | Recomputes | Best For |
|--------|-------|------------|----------|
| `nothing_saveable` | Nothing | Everything | Maximum memory savings |
| `dots_with_no_batch_dims_saveable` | Matmuls | Activations | Transformers (default choice) |
| `everything_saveable` | Everything | Nothing | Debug/comparison only |

### Minimize host-device transfers

PCIe bandwidth (~10-25 GB/s practical) is orders of magnitude slower than GPU HBM bandwidth (~2-3 TB/s).

**✅ DO:**
```python
def training_loop(host_data):
    # Single batch transfer at start
    device_data = jax.device_put(host_data)
    
    for epoch in range(epochs):
        for batch in device_data:  # Data stays on device
            params = train_step(params, batch)
    
    # Transfer only final result
    return jax.device_get(params)
```

**❌ DO NOT:**
```python
def training_loop_bad(host_data):
    for epoch in range(epochs):
        for batch in host_data:
            device_batch = jax.device_put(batch)  # Transfer each batch
            params = train_step(params, device_batch)
            loss = jax.device_get(compute_loss(params, device_batch))  # Transfer back!
            print(f"Loss: {loss}")
```

**Detection tip:** Use `jax.config.update('jax_transfer_guard', 'disallow')` to catch implicit transfers.

---

## Random number generation: The explicit PRNG model

JAX uses explicit PRNG state for reproducibility and parallelizability. Misusing it causes subtle bugs.

### Always split keys—never reuse

**✅ DO:**
```python
import jax.random as random

key = random.key(42)  # Use new-style typed keys (JAX ≥ 0.4.16)

# Split for each random operation
key, subkey = random.split(key)
x = random.normal(subkey, shape=(1000,))

key, subkey = random.split(key)
y = random.uniform(subkey, shape=(1000,))

# For parallel sampling
keys = random.split(key, num=batch_size)
samples = jax.vmap(lambda k: random.normal(k, shape=(100,)))(keys)
```

**❌ DO NOT:**
```python
key = random.key(42)
x = random.normal(key, shape=(100,))
y = random.normal(key, shape=(100,))  # y == x! Same samples!

# Also problematic: potential correlation between different distributions
a = random.uniform(key, shape=(10,))
b = random.normal(key, shape=(10,))  # May be correlated with a!
```

**Why it matters:** JAX PRNGs are deterministic—same key always produces same output. This is a feature for reproducibility, but requires explicit key management. The split pattern ensures independent random streams.

---

## Common anti-patterns: What to avoid

### Dynamic shapes inside traced functions

**❌ DO NOT:**
```python
@jax.jit
def bad_dynamic_shape(length, val):
    return jnp.ones((length,)) * val  # length is traced → shape is dynamic!

@jax.jit
def bad_boolean_indexing(x):
    mask = x > 0
    return x[mask]  # Output size depends on values—undefined at trace time!
```

**✅ DO:**
```python
@jax.jit
def good_masking(x):
    mask = x > 0
    return jnp.where(mask, x, 0)  # Fixed output shape, uses mask for values
```

**Why it matters:** XLA requires static shapes for all intermediate arrays at compile time. Value-dependent shapes can't be determined from tracers.

### Missing block_until_ready in benchmarks

JAX uses asynchronous dispatch—operations return immediately while GPU computes in background.

**❌ DO NOT:**
```python
start = time.time()
result = jax_function(x)  # Returns immediately!
elapsed = time.time() - start  # Measures dispatch (~microseconds), not compute
```

**✅ DO:**
```python
start = time.time()
result = jax_function(x).block_until_ready()  # Wait for GPU
elapsed = time.time() - start  # Actual execution time
```

### Python lists instead of arrays

**❌ DO NOT:**
```python
@jax.jit
def process_list(x):
    return jnp.sum(jnp.array(x))

values = list(range(1000))
result = process_list(values)  # Each element becomes a separate tracer!
```

**✅ DO:**
```python
@jax.jit
def process_array(x):
    return jnp.sum(x)

values = jnp.arange(1000)  # Convert BEFORE JIT boundary
result = process_array(values)
```

### JIT function defined inside loops

**❌ DO NOT:**
```python
def train_loop():
    for epoch in range(100):
        @jax.jit  # New function object each iteration!
        def step(x):
            return x * 2
        result = step(data)  # Recompiles every iteration
```

**✅ DO:**
```python
@jax.jit  # Define ONCE, outside loop
def step(x):
    return x * 2

def train_loop():
    for epoch in range(100):
        result = step(data)  # Reuses cached compilation
```

---

## Debugging and profiling tools

### Essential configuration flags

```python
# Log every compilation (catch recompilation issues)
jax.config.update('jax_log_compiles', True)

# Explain why cache misses occur
jax.config.update('jax_explain_cache_misses', True)

# Catch implicit host-device transfers
jax.config.update('jax_transfer_guard', 'disallow')

# Enable persistent compilation cache (saves across restarts)
jax.config.update('jax_compilation_cache_dir', '/path/to/cache')
```

### Runtime debugging in JIT functions

**✅ DO:**
```python
@jax.jit
def debugged_function(x):
    y = x * 2
    jax.debug.print("y = {}", y)  # Prints ACTUAL values at runtime
    return y + 1
```

**❌ DO NOT:**
```python
@jax.jit
def bad_debug(x):
    print(f"x = {x}")  # Prints "Traced<ShapedArray...>", only at trace time!
    return x * 2
```

### Memory profiling

```python
# Capture device memory profile
result.block_until_ready()
jax.profiler.save_device_memory_profile("memory.prof")
# Visualize: pprof --web memory.prof

# Execution trace for TensorBoard/Perfetto
with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    result = computation(x)
    result.block_until_ready()
```

---

## Hardware-specific considerations

| Aspect | CPU | GPU | TPU |
|--------|-----|-----|-----|
| **JIT benefit** | Moderate | High (10-100x) | High |
| **vmap speedup** | 2-5x | 10-100x | 10-100x |
| **Compilation time** | Fast | Medium | Slow (minutes) |
| **Optimal precision** | float32 | float32/float16 | bfloat16 |
| **Memory prealloc** | N/A | 75% default | Managed |
| **Buffer donation** | Limited benefit | Significant | Critical |
| **Dispatch overhead** | Low | Medium | Higher |

### GPU-specific optimizations

```python
import os
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_enable_latency_hiding_scheduler=true '  # Overlap comm/compute
    '--xla_gpu_triton_gemm_any=True '  # Use Triton GEMM when beneficial
)
```

### TPU-specific optimizations

- Align batch/head dimensions to **256** for TPU v4/v5/v6 systolic arrays
- Use `bfloat16` by default—it's native precision
- Avoid dynamic shapes entirely—TPU compilation is expensive

---

## Quick reference: The 10 commandments of JAX performance

1. **JIT everything** — Wrap your entire training step, not individual operations
2. **Keep data on device** — Transfer once at start, once at end
3. **Use fixed shapes** — Pad or bucket to avoid recompilation
4. **Use JAX control flow** — `lax.cond`, `lax.scan`, not Python `if`/`for`
5. **Split PRNG keys** — Never reuse, always split
6. **Donate buffers** — Use `donate_argnums` in training loops
7. **Checkpoint deep networks** — Trade compute for memory
8. **vmap, don't loop** — Vectorize over batch dimensions
9. **block_until_ready** — Always block before timing
10. **Profile before optimizing** — Use `jax.profiler` with TensorBoard

---

## Detection cheat sheet

| Symptom | Likely Cause | Detection |
|---------|--------------|-----------|
| First call slow, rest fast | Normal JIT compilation | Expected behavior—warmup |
| Every call slow | Recompilation | `jax_log_compiles=True` |
| OOM on forward pass | No checkpointing | Memory profiler |
| OOM on backward pass | Large intermediates | `jax.checkpoint` layers |
| Wrong random values | Key reuse | Review PRNG key management |
| TracerBoolConversionError | Python control flow | Use `lax.cond` |
| Traced<ShapedArray> in output | print() in JIT | Use `jax.debug.print()` |
| Suspiciously fast benchmarks | Missing block_until_ready | Add `.block_until_ready()` |

---

*This cheatsheet synthesizes patterns from official JAX documentation, JAX GitHub discussions, Google's XLA optimization guides, and JAX-based libraries (Flax, Optax) as of January 2025.*