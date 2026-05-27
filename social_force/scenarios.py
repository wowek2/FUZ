"""Tutaj definiujemy scenariusze symulacji"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .model import ModelParameters, PedestrianState, Wall
from .simulation import Simulator


class Scenario(Protocol):
    """Interfejs wspoldzielony dla scenariuszy."""

    name: str

    def build(self, rng: np.random.Generator) -> Simulator:
        """Buduje symulator."""
        ...

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Granice rysowania"""
        ...


def _sample_desired_speeds(
    n: int, params: ModelParameters, rng: np.random.Generator
) -> np.ndarray:
    """Losyjemy preferowane prędkości z rozkładu N(v0_mean, v0_std)."""
    speeds = rng.normal(params.v0_mean, params.v0_std, size=n)
    return np.maximum(speeds, 0.1)


def _scatter_positions(
    n: int,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    min_distance: float,
    rng: np.random.Generator,
    max_tries: int = 5000,
) -> np.ndarray:
    """Rozmieszcza ``n`` pozycji w prostokącie z zachowaniem minimalnego dystansu.

    Jeśli nie da sie zachowac dystansu to dalsze pozycje rozmieszczane losowo bez ograniczen.
    """
    placed: list[np.ndarray] = []
    tries = 0
    while len(placed) < n and tries < max_tries:
        candidate = np.array([rng.uniform(*x_range), rng.uniform(*y_range)])
        if all(np.linalg.norm(candidate - p) >= min_distance for p in placed):
            placed.append(candidate)
        tries += 1
    while len(placed) < n:
        placed.append(np.array([rng.uniform(*x_range), rng.uniform(*y_range)]))
    return np.array(placed)


@dataclass
class LaneFormationScenario:
    """
    Dwie grupy idace w przeciwnych kierunkachh
    """

    name: str = "lanes"
    length: float = 30.0
    width: float = 10.0
    n_pedestrians: int = 80
    params: ModelParameters = field(default_factory=ModelParameters)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (0.0, self.length, -1.0, self.width + 1.0)

    def _walls(self) -> list[Wall]:
        return [
            Wall(a=np.array([0.0, 0.0]), b=np.array([self.length, 0.0])),
            Wall(
                a=np.array([0.0, self.width]),
                b=np.array([self.length, self.width]),
            ),
        ]

    def build(self, rng: np.random.Generator) -> Simulator:
        n = self.n_pedestrians
        margin = 0.5
        positions = _scatter_positions(
            n,
            (margin, self.length - margin),
            (margin, self.width - margin),
            min_distance=0.6,
            rng=rng,
        )
        group = np.zeros(n, dtype=int)
        group[n // 2 :] = 1
        desired_speed = _sample_desired_speeds(n, self.params, rng)

        # Grupa 0
        directions = np.where(group == 0, 1.0, -1.0)
        velocities = np.zeros((n, 2))
        velocities[:, 0] = directions * desired_speed

        # Cel w +inf
        far = self.length * 100.0
        target = np.column_stack(
            [np.where(group == 0, far, -far), positions[:, 1]]
        )

        state = PedestrianState(
            positions=positions,
            velocities=velocities,
            desired_speed=desired_speed,
            target=target,
            group=group,
        )

        def maintain(s: PedestrianState, _dt: float) -> None:
            s.positions[:, 0] = s.positions[:, 0] % self.length
            s.target[:, 1] = s.positions[:, 1]

        return Simulator(
            state=state,
            walls=self._walls(),
            params=self.params,
            maintain=maintain,
        )


@dataclass
class BottleneckScenario:
    """
    Dwie grupy i drzwi
    """

    name: str = "bottleneck"
    length: float = 20.0
    width: float = 10.0
    door_width: float = 2.0
    n_per_group: int = 15
    params: ModelParameters = field(default_factory=ModelParameters)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (0.0, self.length, -1.0, self.width + 1.0)

    def _walls(self) -> list[Wall]:
        x_mid = self.length / 2.0
        half_door = self.door_width / 2.0
        y_mid = self.width / 2.0
        return [
            Wall(a=np.array([0.0, 0.0]), b=np.array([self.length, 0.0])),
            Wall(
                a=np.array([0.0, self.width]),
                b=np.array([self.length, self.width]),
            ),
            Wall(
                a=np.array([x_mid, 0.0]),
                b=np.array([x_mid, y_mid - half_door]),
            ),
            Wall(
                a=np.array([x_mid, y_mid + half_door]),
                b=np.array([x_mid, self.width]),
            ),
        ]

    def build(self, rng: np.random.Generator) -> Simulator:
        n = 2 * self.n_per_group
        margin = 0.5
        x_mid = self.length / 2.0
        wall_clear = 0.4

        left = _scatter_positions(
            self.n_per_group,
            (margin, x_mid - wall_clear),
            (margin, self.width - margin),
            min_distance=0.6,
            rng=rng,
        )
        right = _scatter_positions(
            self.n_per_group,
            (x_mid + wall_clear, self.length - margin),
            (margin, self.width - margin),
            min_distance=0.6,
            rng=rng,
        )
        positions = np.vstack([left, right])

        group = np.zeros(n, dtype=int)
        group[self.n_per_group :] = 1

        desired_speed = _sample_desired_speeds(n, self.params, rng)
        directions = np.where(group == 0, 1.0, -1.0)
        velocities = np.zeros((n, 2))
        velocities[:, 0] = 0.5 * directions * desired_speed

        # Poczatkowa inicjalizacja celu, nadpisywany w callbac maintain.
        target = np.column_stack(
            [np.full(n, x_mid), np.full(n, self.width / 2.0)]
        )

        state = PedestrianState(
            positions=positions,
            velocities=velocities,
            desired_speed=desired_speed,
            target=target,
            group=group,
        )

        length = self.length
        half_door = self.door_width / 2.0
        y_mid = self.width / 2.0
        door_buffer = 0.15
        door_y_lo = y_mid - half_door + door_buffer
        door_y_hi = y_mid + half_door - door_buffer
        cross_eps = 0.5

        def maintain(s: PedestrianState, _dt: float) -> None:
            x = s.positions[:, 0]
            right_movers = s.group == 0
            left_movers = s.group == 1

            on_start_right = right_movers & (x < x_mid)
            passed_right = right_movers & (x >= x_mid)
            on_start_left = left_movers & (x > x_mid)
            passed_left = left_movers & (x <= x_mid)

            door_y = np.clip(s.positions[:, 1], door_y_lo, door_y_hi)

            s.target[on_start_right, 0] = x_mid + cross_eps
            s.target[on_start_right, 1] = door_y[on_start_right]
            s.target[passed_right, 0] = length + 5.0
            s.target[passed_right, 1] = y_mid

            s.target[on_start_left, 0] = x_mid - cross_eps
            s.target[on_start_left, 1] = door_y[on_start_left]
            s.target[passed_left, 0] = -5.0
            s.target[passed_left, 1] = y_mid

            # Pieszy po wyjsciu zzawuja na poczatek
            exited_right = right_movers & (x > length - 0.1)
            exited_left = left_movers & (x < 0.1)
            if np.any(exited_right):
                s.positions[exited_right, 0] = 0.2
                s.positions[exited_right, 1] = rng.uniform(
                    margin, self.width - margin, size=int(exited_right.sum())
                )
                s.velocities[exited_right] *= 0.0
            if np.any(exited_left):
                s.positions[exited_left, 0] = length - 0.2
                s.positions[exited_left, 1] = rng.uniform(
                    margin, self.width - margin, size=int(exited_left.sum())
                )
                s.velocities[exited_left] *= 0.0

        return Simulator(
            state=state,
            walls=self._walls(),
            params=self.params,
            maintain=maintain,
        )
