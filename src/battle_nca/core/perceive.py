"""Perception layers for NCA using Sobel gradients and depthwise convolutions."""

import jax
import jax.numpy as jnp
from flax import linen as nn
from functools import partial


def _create_sobel_kernels() -> tuple[jnp.ndarray, jnp.ndarray]:
    """Create Sobel kernels for gradient estimation."""
    sobel_x = jnp.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=jnp.float32)
    sobel_y = sobel_x.T
    return sobel_x, sobel_y


def circular_pad(x: jnp.ndarray, pad: int = 1) -> jnp.ndarray:
    """Apply circular (wrap-around) padding for toroidal topology.

    Args:
        x: Input array of shape (H, W, C) or (B, H, W, C)
        pad: Padding size

    Returns:
        Padded array
    """
    if x.ndim == 3:
        return jnp.pad(x, ((pad, pad), (pad, pad), (0, 0)), mode='wrap')
    return jnp.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode='wrap')


def depthwise_conv(
    inputs: jnp.ndarray,
    kernel: jnp.ndarray,
    channels: int,
    use_circular_padding: bool = True
) -> jnp.ndarray:
    """Apply depthwise convolution with optional circular padding.

    Args:
        inputs: Input tensor of shape (H, W, C) or (B, H, W, C)
        kernel: Convolution kernel of shape (K, K)
        channels: Number of channels
        use_circular_padding: Whether to use circular padding

    Returns:
        Convolved tensor
    """
    has_batch = inputs.ndim == 4
    if not has_batch:
        inputs = inputs[None]

    kernel_size = kernel.shape[0]
    pad_size = kernel_size // 2

    if use_circular_padding:
        inputs = circular_pad(inputs, pad=pad_size)
        padding = 'VALID'
    else:
        padding = 'SAME'

    # For depthwise conv with feature_group_count=channels:
    # kernel shape should be (H, W, 1, channels) in HWIO format
    kernel_expanded = kernel[:, :, None, None]
    kernel_tiled = jnp.tile(kernel_expanded, (1, 1, 1, channels))

    result = jax.lax.conv_general_dilated(
        inputs,
        kernel_tiled,
        (1, 1),
        padding,
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        feature_group_count=channels
    )

    if not has_batch:
        result = result[0]
    return result


def perceive(
    state: jnp.ndarray,
    use_circular_padding: bool = True
) -> jnp.ndarray:
    """Compute perception vector using Sobel gradients + identity.

    This creates a 3x perception vector: [state, grad_x, grad_y]

    Args:
        state: Cell state of shape (H, W, C) or (B, H, W, C)
        use_circular_padding: Whether to use circular padding

    Returns:
        Perception tensor of shape (..., H, W, 3*C)
    """
    sobel_x, sobel_y = _create_sobel_kernels()
    channels = state.shape[-1]

    grad_x = depthwise_conv(state, sobel_x, channels, use_circular_padding)
    grad_y = depthwise_conv(state, sobel_y, channels, use_circular_padding)

    return jnp.concatenate([state, grad_x, grad_y], axis=-1)


class DepthwiseConvPerceive(nn.Module):
    """Flax module for perception with depthwise convolutions.

    Attributes:
        num_channels: Number of input channels
        kernel_size: Convolution kernel size (default 3)
        use_circular_padding: Whether to use circular padding
        include_self: Whether to include identity in perception
    """
    num_channels: int
    kernel_size: int = 3
    use_circular_padding: bool = True
    include_self: bool = True

    @nn.compact
    def __call__(self, state: jnp.ndarray) -> jnp.ndarray:
        """Compute perception vector.

        Args:
            state: Input state of shape (H, W, C) or (B, H, W, C)

        Returns:
            Perception tensor
        """
        sobel_x, sobel_y = _create_sobel_kernels()

        grad_x = depthwise_conv(
            state, sobel_x, self.num_channels, self.use_circular_padding
        )
        grad_y = depthwise_conv(
            state, sobel_y, self.num_channels, self.use_circular_padding
        )

        if self.include_self:
            return jnp.concatenate([state, grad_x, grad_y], axis=-1)
        return jnp.concatenate([grad_x, grad_y], axis=-1)


def _create_gaussian_kernel(size: int, sigma: float) -> jnp.ndarray:
    """Create a Gaussian kernel for smoothing (module-level for caching)."""
    x = jnp.arange(size) - size // 2
    xx, yy = jnp.meshgrid(x, x)
    kernel = jnp.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / kernel.sum()


# Pre-compute Gaussian kernels at module load time (they never change)
_MORALE_KERNEL = _create_gaussian_kernel(7, 2.0)
_FORMATION_KERNEL = _create_gaussian_kernel(11, 4.0)


class MultiScalePerceive(nn.Module):
    """Perception at multiple spatial scales for different mechanics.

    Uses different perception radii:
    - 3x3 for melee combat
    - 7x7 for morale contagion
    - 11x11 for formation cohesion

    OPTIMIZED: Gaussian kernels are pre-computed at module load time.

    Attributes:
        num_channels: Number of input channels
        use_circular_padding: Whether to use circular padding
    """
    num_channels: int
    use_circular_padding: bool = True

    @nn.compact
    def __call__(self, state: jnp.ndarray) -> dict[str, jnp.ndarray]:
        """Compute multi-scale perception.

        Args:
            state: Input state of shape (H, W, C) or (B, H, W, C)

        Returns:
            Dictionary with 'melee', 'morale', 'formation' perception tensors
        """
        has_batch = state.ndim == 4
        if not has_batch:
            state = state[None]

        # Standard Sobel perception for melee (3x3)
        melee_perception = perceive(state, self.use_circular_padding)

        # Morale perception (7x7 Gaussian smoothing) - use pre-computed kernel
        morale_smooth = depthwise_conv(
            state, _MORALE_KERNEL, self.num_channels, self.use_circular_padding
        )

        # Formation perception (11x11 Gaussian smoothing) - use pre-computed kernel
        formation_smooth = depthwise_conv(
            state, _FORMATION_KERNEL, self.num_channels, self.use_circular_padding
        )

        if not has_batch:
            melee_perception = melee_perception[0]
            morale_smooth = morale_smooth[0]
            formation_smooth = formation_smooth[0]

        return {
            'melee': melee_perception,
            'morale': morale_smooth,
            'formation': formation_smooth
        }
