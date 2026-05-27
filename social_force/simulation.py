from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .forces import total_force
from .model import ModelParameters, PedestrianState, Wall

MaintainHook = Callable[[PedestrianState, float], None]


@dataclass
class Simulator:
    """
    Symulator ruchu pieszych w modelu
    """

    state: PedestrianState
    walls: list[Wall]
    params: ModelParameters
    maintain: Optional[MaintainHook] = None
    time: float = 0.0
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    def step(self) -> None:
        """
        Wykonuje jeden krok całkowania (dlugosc zdefiniowana przez params.dt)
        """
        force = total_force(self.state, self.walls, self.params, self.rng)

        preferred = self.state.velocities + self.params.dt * force

        v_max = self.params.v_max_factor * self.state.desired_speed
        preferred_norm = np.linalg.norm(preferred, axis=-1)
        scale = np.minimum(1.0, v_max / np.maximum(preferred_norm, 1e-12))
        actual = preferred * scale[:, None]

        self.state.velocities = actual
        self.state.positions = self.state.positions + self.params.dt * actual
        self.time += self.params.dt

        if self.maintain is not None:
            self.maintain(self.state, self.params.dt)

    def run(self, duration: float) -> None:
        """Odpalamy krki w symulacji az do konca czasu (duration)"""
        n_steps = int(np.ceil(duration / self.params.dt))
        for _ in range(n_steps):
            self.step()
