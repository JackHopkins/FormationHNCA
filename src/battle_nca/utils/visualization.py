"""Visualization utilities for battle NCA."""

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from typing import Optional
import io


# Custom colormaps
MORALE_CMAP = LinearSegmentedColormap.from_list(
    'morale', ['red', 'yellow', 'green']
)
HEALTH_CMAP = LinearSegmentedColormap.from_list(
    'health', ['black', 'red', 'green']
)


def render_state(
    state: jnp.ndarray,
    mode: str = 'rgba',
    channel: int | None = None,
    cmap: str = 'viridis',
    title: str | None = None,
    ax: plt.Axes | None = None,
    show: bool = True
) -> plt.Figure | None:
    """Render NCA state as an image.

    Args:
        state: State tensor (H, W, C)
        mode: Rendering mode - 'rgba', 'rgb', 'alpha', 'channel', 'health', 'morale'
        channel: Channel index for 'channel' mode
        cmap: Colormap for single-channel modes
        title: Optional title
        ax: Optional axes to draw on
        show: Whether to display the figure

    Returns:
        Figure if show=False, else None
    """
    state = np.array(state)  # Convert from JAX

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure

    if mode == 'rgba':
        # RGBA visualization
        img = np.clip(state[..., :4], 0, 1)
        # Premultiply alpha for display
        rgb = img[..., :3]
        alpha = img[..., 3:4]
        # White background
        img_display = rgb * alpha + (1 - alpha)
        ax.imshow(img_display)

    elif mode == 'rgb':
        img = np.clip(state[..., :3], 0, 1)
        ax.imshow(img)

    elif mode == 'alpha':
        ax.imshow(state[..., 3], cmap='gray', vmin=0, vmax=1)

    elif mode == 'channel':
        if channel is None:
            raise ValueError("channel must be specified for mode='channel'")
        ax.imshow(state[..., channel], cmap=cmap)
        plt.colorbar(ax.images[0], ax=ax)

    elif mode == 'health':
        ax.imshow(state[..., 4], cmap=HEALTH_CMAP, vmin=0, vmax=1)
        plt.colorbar(ax.images[0], ax=ax, label='Health')

    elif mode == 'morale':
        ax.imshow(state[..., 5], cmap=MORALE_CMAP, vmin=-1, vmax=1)
        plt.colorbar(ax.images[0], ax=ax, label='Morale')

    else:
        raise ValueError(f"Unknown mode: {mode}")

    ax.axis('off')
    if title:
        ax.set_title(title)

    if show:
        plt.show()
        return None
    return fig


def render_battle(
    red_state: jnp.ndarray,
    blue_state: jnp.ndarray,
    figsize: tuple[int, int] = (16, 8),
    title: str = 'Battle State',
    show: bool = True
) -> plt.Figure | None:
    """Render two-army battle state.

    Args:
        red_state: Red army state (H, W, C)
        blue_state: Blue army state (H, W, C)
        figsize: Figure size
        title: Figure title
        show: Whether to display

    Returns:
        Figure if show=False
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle(title)

    red_state = np.array(red_state)
    blue_state = np.array(blue_state)

    # Top row: Red army
    axes[0, 0].imshow(np.clip(red_state[..., :4], 0, 1))
    axes[0, 0].set_title('Red Army - RGBA')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(red_state[..., 4], cmap=HEALTH_CMAP, vmin=0, vmax=1)
    axes[0, 1].set_title('Red Army - Health')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(red_state[..., 5], cmap=MORALE_CMAP, vmin=-1, vmax=1)
    axes[0, 2].set_title('Red Army - Morale')
    axes[0, 2].axis('off')

    # Bottom row: Blue army
    axes[1, 0].imshow(np.clip(blue_state[..., :4], 0, 1))
    axes[1, 0].set_title('Blue Army - RGBA')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(blue_state[..., 4], cmap=HEALTH_CMAP, vmin=0, vmax=1)
    axes[1, 1].set_title('Blue Army - Health')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(blue_state[..., 5], cmap=MORALE_CMAP, vmin=-1, vmax=1)
    axes[1, 2].set_title('Blue Army - Morale')
    axes[1, 2].axis('off')

    plt.tight_layout()

    if show:
        plt.show()
        return None
    return fig


def create_animation(
    trajectory: jnp.ndarray,
    mode: str = 'rgba',
    fps: int = 30,
    figsize: tuple[int, int] = (8, 8),
    title: str = 'NCA Evolution'
) -> animation.FuncAnimation:
    """Create animation from NCA trajectory.

    Args:
        trajectory: Trajectory tensor (T, H, W, C)
        mode: Rendering mode
        fps: Frames per second
        figsize: Figure size
        title: Animation title

    Returns:
        Matplotlib animation
    """
    trajectory = np.array(trajectory)
    num_frames = len(trajectory)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title)
    ax.axis('off')

    # Initialize with first frame
    if mode == 'rgba':
        img_data = np.clip(trajectory[0, ..., :4], 0, 1)
        rgb = img_data[..., :3]
        alpha = img_data[..., 3:4]
        display = rgb * alpha + (1 - alpha)
        im = ax.imshow(display)
    elif mode == 'alpha':
        im = ax.imshow(trajectory[0, ..., 3], cmap='gray', vmin=0, vmax=1)
    elif mode == 'health':
        im = ax.imshow(trajectory[0, ..., 4], cmap=HEALTH_CMAP, vmin=0, vmax=1)
    elif mode == 'morale':
        im = ax.imshow(trajectory[0, ..., 5], cmap=MORALE_CMAP, vmin=-1, vmax=1)
    else:
        im = ax.imshow(trajectory[0, ..., 0], cmap='viridis')

    frame_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                         verticalalignment='top', fontsize=10,
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    def update(frame):
        if mode == 'rgba':
            img_data = np.clip(trajectory[frame, ..., :4], 0, 1)
            rgb = img_data[..., :3]
            alpha = img_data[..., 3:4]
            display = rgb * alpha + (1 - alpha)
            im.set_array(display)
        elif mode == 'alpha':
            im.set_array(trajectory[frame, ..., 3])
        elif mode == 'health':
            im.set_array(trajectory[frame, ..., 4])
        elif mode == 'morale':
            im.set_array(trajectory[frame, ..., 5])
        else:
            im.set_array(trajectory[frame, ..., 0])

        frame_text.set_text(f'Frame: {frame}/{num_frames-1}')
        return [im, frame_text]

    anim = animation.FuncAnimation(
        fig, update, frames=num_frames,
        interval=1000 // fps, blit=True
    )

    return anim


def create_battle_animation(
    red_trajectory: jnp.ndarray,
    blue_trajectory: jnp.ndarray,
    fps: int = 30,
    figsize: tuple[int, int] = (16, 8)
) -> animation.FuncAnimation:
    """Create animation of two-army battle.

    Args:
        red_trajectory: Red army trajectory (T, H, W, C)
        blue_trajectory: Blue army trajectory (T, H, W, C)
        fps: Frames per second
        figsize: Figure size

    Returns:
        Matplotlib animation
    """
    red = np.array(red_trajectory)
    blue = np.array(blue_trajectory)
    num_frames = min(len(red), len(blue))

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].set_title('Red Army')
    axes[1].set_title('Blue Army')

    for ax in axes:
        ax.axis('off')

    # Initialize
    def get_display(state):
        img = np.clip(state[..., :4], 0, 1)
        rgb = img[..., :3]
        alpha = img[..., 3:4]
        return rgb * alpha + (1 - alpha)

    im_red = axes[0].imshow(get_display(red[0]))
    im_blue = axes[1].imshow(get_display(blue[0]))

    frame_text = fig.text(0.5, 0.02, '', ha='center', fontsize=12)

    def update(frame):
        im_red.set_array(get_display(red[frame]))
        im_blue.set_array(get_display(blue[frame]))
        frame_text.set_text(f'Step: {frame}/{num_frames-1}')
        return [im_red, im_blue, frame_text]

    anim = animation.FuncAnimation(
        fig, update, frames=num_frames,
        interval=1000 // fps, blit=True
    )

    return anim


def plot_training_curves(
    metrics: dict[str, list[float]],
    figsize: tuple[int, int] = (12, 4),
    title: str = 'Training Curves',
    show: bool = True
) -> plt.Figure | None:
    """Plot training loss curves.

    Args:
        metrics: Dictionary with 'losses' and optionally other metrics
        figsize: Figure size
        title: Plot title
        show: Whether to display

    Returns:
        Figure if show=False
    """
    num_plots = len(metrics)
    fig, axes = plt.subplots(1, num_plots, figsize=figsize)

    if num_plots == 1:
        axes = [axes]

    for ax, (name, values) in zip(axes, metrics.items()):
        ax.plot(values)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(name.capitalize())
        ax.set_title(f'{name.capitalize()} over Training')

        # Add smoothed line
        if len(values) > 20:
            window = min(50, len(values) // 10)
            smoothed = np.convolve(values, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(values)), smoothed, 'r-', alpha=0.7,
                    label='Smoothed')
            ax.legend()

    fig.suptitle(title)
    plt.tight_layout()

    if show:
        plt.show()
        return None
    return fig


def visualize_channels(
    state: jnp.ndarray,
    channels: list[int] | None = None,
    channel_names: list[str] | None = None,
    figsize: tuple[int, int] = (16, 8),
    show: bool = True
) -> plt.Figure | None:
    """Visualize multiple channels of NCA state.

    Args:
        state: State tensor (H, W, C)
        channels: List of channel indices to show (default: first 12)
        channel_names: Optional names for channels
        figsize: Figure size
        show: Whether to display

    Returns:
        Figure if show=False
    """
    state = np.array(state)

    if channels is None:
        channels = list(range(min(12, state.shape[-1])))

    num_channels = len(channels)
    cols = min(4, num_channels)
    rows = (num_channels + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.atleast_2d(axes)

    default_names = [
        'R', 'G', 'B', 'Alpha',
        'Health', 'Morale', 'Fatigue',
        'Vel_X', 'Vel_Y',
        'Type', 'Formation',
        'Parent_0', 'Parent_1',
        'Enemy_Prox', 'Enemy_Dir'
    ]

    for i, ch in enumerate(channels):
        row, col = i // cols, i % cols
        ax = axes[row, col]

        im = ax.imshow(state[..., ch], cmap='viridis')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        if channel_names and i < len(channel_names):
            name = channel_names[i]
        elif ch < len(default_names):
            name = default_names[ch]
        else:
            name = f'Channel {ch}'

        ax.set_title(name)
        ax.axis('off')

    # Hide unused axes
    for i in range(num_channels, rows * cols):
        row, col = i // cols, i % cols
        axes[row, col].axis('off')

    plt.tight_layout()

    if show:
        plt.show()
        return None
    return fig


def save_animation(
    anim: animation.FuncAnimation,
    filename: str,
    fps: int = 30,
    dpi: int = 100
) -> None:
    """Save animation to file.

    Args:
        anim: Animation to save
        filename: Output filename (supports .mp4, .gif)
        fps: Frames per second
        dpi: Resolution
    """
    if filename.endswith('.gif'):
        writer = animation.PillowWriter(fps=fps)
    else:
        writer = animation.FFMpegWriter(fps=fps)

    anim.save(filename, writer=writer, dpi=dpi)
    print(f"Animation saved to {filename}")


def fig_to_array(fig: plt.Figure) -> np.ndarray:
    """Convert matplotlib figure to numpy array.

    Args:
        fig: Matplotlib figure

    Returns:
        RGB array
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)

    from PIL import Image
    img = Image.open(buf)
    return np.array(img)
