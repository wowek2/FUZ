from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection

from social_force import Scenario, Simulator


_GROUP_COLORS = ["tab:blue", "tab:orange"]


@dataclass
class AnimationConfig:
    """
    Parametry rysowania animacji.
    """

    substeps: int = 15
    interval_ms: int = 30
    trail_length: int = 25
    show_velocity_arrows: bool = True
    marker_size: float = 60.0


@dataclass
class PedestrianAnimator:
    """Buduje i uruchamia animację.
    """

    simulator: Simulator
    scenario: Scenario
    config: AnimationConfig = field(default_factory=AnimationConfig)
    max_seconds: Optional[float] = None

    def _setup_axes(self) -> tuple[plt.Figure, plt.Axes]:
        x_min, x_max, y_min, y_max = self.scenario.bounds
        fig_width = 12.0
        aspect = (y_max - y_min) / max(x_max - x_min, 1e-6)
        fig, ax = plt.subplots(
            figsize=(fig_width, max(3.0, fig_width * aspect + 1.2))
        )
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(f"Model sil - {self.scenario.name}")
        ax.grid(True, linewidth=0.5, alpha=0.3)
        return fig, ax

    def _draw_walls(self, ax: plt.Axes) -> None:
        for wall in self.simulator.walls:
            ax.plot(
                [wall.a[0], wall.b[0]],
                [wall.a[1], wall.b[1]],
                color="black",
                linewidth=3.0,
                zorder=1,
            )

    def animate(self) -> FuncAnimation:
        """Buduje obiekt FuncAnimation."""
        fig, ax = self._setup_axes()
        self._draw_walls(ax)

        state = self.simulator.state
        n = state.n
        colors = np.array(
            [_GROUP_COLORS[g % len(_GROUP_COLORS)] for g in state.group]
        )

        scatter = ax.scatter(
            state.positions[:, 0],
            state.positions[:, 1],
            c=colors,
            s=self.config.marker_size,
            edgecolors="black",
            linewidths=0.5,
            zorder=3,
        )

        quiver = (
            ax.quiver(
                state.positions[:, 0],
                state.positions[:, 1],
                state.velocities[:, 0],
                state.velocities[:, 1],
                color="gray",
                alpha=0.6,
                scale=25.0,
                width=0.0025,
                zorder=2,
            )
            if self.config.show_velocity_arrows
            else None
        )

        trail_history: deque[np.ndarray] = deque(maxlen=self.config.trail_length)
        trail_lines = LineCollection([], linewidths=1.0, alpha=0.35, zorder=2)
        ax.add_collection(trail_lines)

        time_text = ax.text(
            0.01, 0.97, "", transform=ax.transAxes, verticalalignment="top"
        )

        x_min, x_max, _, _ = self.scenario.bounds
        x_range = x_max - x_min

        def update(_frame: int) -> list:
            for _ in range(self.config.substeps):
                self.simulator.step()

            positions = state.positions
            velocities = state.velocities

            scatter.set_offsets(positions)

            if quiver is not None:
                quiver.set_offsets(positions)
                quiver.set_UVC(velocities[:, 0], velocities[:, 1])

            trail_history.append(positions.copy())
            if len(trail_history) >= 2:
                _update_trails(trail_history, trail_lines, colors, x_range)

            time_text.set_text(f"t = {self.simulator.time:6.2f} s")

            artists: list = [scatter, trail_lines, time_text]
            if quiver is not None:
                artists.append(quiver)
            return artists

        frames = None
        if self.max_seconds is not None:
            frames = int(
                self.max_seconds
                / (self.simulator.params.dt * self.config.substeps)
            )

        return FuncAnimation(
            fig,
            update,
            frames=frames,
            interval=self.config.interval_ms,
            blit=False,
            cache_frame_data=False,
        )


def _update_trails(
    history: deque[np.ndarray],
    trail_lines: LineCollection,
    colors: np.ndarray,
    x_range: float,
) -> None:
    h = np.stack(history)  # (T, N, 2)
    deltas = np.linalg.norm(np.diff(h, axis=0), axis=-1)  # (T-1, N)
    # Segmenty z przeskokiem dłuższym niż połowa korytarza traktujemy
    # jako zawinięcie i pomijamy.
    valid = deltas <= x_range / 2.0
    segments = np.stack([h[:-1], h[1:]], axis=2)  # (T-1, N, 2, 2)

    n = h.shape[1]
    seg_colors = np.broadcast_to(colors, (h.shape[0] - 1, n))

    trail_lines.set_segments(segments[valid])
    trail_lines.set_color(seg_colors[valid])


def save_animation(anim: FuncAnimation, path: str, fps: int = 30) -> None:
    """
    Zapisuje animację do pliku.
    """
    if path.lower().endswith(".gif"):
        anim.save(path, writer=PillowWriter(fps=fps))
    else:
        anim.save(path, fps=fps)
