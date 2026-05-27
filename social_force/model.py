"""Struktury danych"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ModelParameters:
    """Parametry modelu"""

    tau: float = 0.5
    V0: float = 2.1
    sigma: float = 0.3
    U0: float = 10.0
    R: float = 0.2
    delta_t: float = 2.0
    phi: float = np.deg2rad(100.0)
    c: float = 0.5
    v_max_factor: float = 1.3
    v0_mean: float = 1.34
    v0_std: float = 0.26
    dt: float = 0.01
    noise_std: float = 0.0
    radius: float = 0.25


@dataclass
class PedestrianState:
    """
    Zwektoryzowany stan N pieszych.
    """

    positions: np.ndarray
    velocities: np.ndarray
    desired_speed: np.ndarray
    target: np.ndarray
    group: np.ndarray

    @property
    def n(self) -> int:
        """Liczba pieszych."""
        return int(self.positions.shape[0])


@dataclass
class Wall:
    """Odcinek ściany od punktu a do b."""

    a: np.ndarray
    b: np.ndarray

    def nearest_point(self, p: np.ndarray) -> np.ndarray:
        ab = self.b - self.a
        ab_dot = float(ab @ ab)
        if ab_dot < 1e-12:
            return np.broadcast_to(self.a, p.shape).copy()
        t = np.clip((p - self.a) @ ab / ab_dot, 0.0, 1.0)
        return self.a + np.outer(t, ab)
