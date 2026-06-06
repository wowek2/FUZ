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

@dataclass
class CornerScenario:
    """
    Korytarz z zakrętem pod kątem 90 stopni (L-shape).
    Dopasowane proporcje okna.
    """

    name: str = "corner"
    corridor_width: float = 10.0
    leg1_length: float = 20.0
    leg2_length: float = 10.0  # Skrócone, by okno przypominało szeroki prostokąt
    n_pedestrians: int = 60
    params: ModelParameters = field(default_factory=ModelParameters)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        # Dodane marginesy -1.0 i +1.0, żeby kamera obejmowała nieco przestrzeni poza ścianami
        return (
            -1.0,
            self.leg1_length + self.corridor_width + 1.0,
            -1.0,
            self.leg2_length + self.corridor_width + 1.0,
        )

    def _walls(self) -> list[Wall]:
        w = self.corridor_width
        l1 = self.leg1_length
        l2 = self.leg2_length
        return [
            # Zewnętrzna ściana (dół i prawa strona zakrętu)
            Wall(a=np.array([0.0, 0.0]), b=np.array([l1 + w, 0.0])),
            Wall(a=np.array([l1 + w, 0.0]), b=np.array([l1 + w, w + l2])),
            # Wewnętrzna ściana (góra i lewa strona zakrętu)
            Wall(a=np.array([0.0, w]), b=np.array([l1, w])),
            Wall(a=np.array([l1, w]), b=np.array([l1, w + l2])),
        ]

    def build(self, rng: np.random.Generator) -> Simulator:
        n = self.n_pedestrians
        margin = 0.5
        w = self.corridor_width
        l1 = self.leg1_length
        l2 = self.leg2_length

        # Agenci startują na początku pierwszego korytarza (po lewej)
        positions = _scatter_positions(
            n,
            (margin, 5.0),
            (margin, w - margin),
            min_distance=0.6,
            rng=rng,
        )

        group = np.zeros(n, dtype=int)
        desired_speed = _sample_desired_speeds(n, self.params, rng)
        velocities = np.zeros((n, 2))
        velocities[:, 0] = desired_speed

        # Początkowy cel: środek zakrętu
        target = np.column_stack([
            np.full(n, l1 + w / 2.0),
            np.full(n, w / 2.0)
        ])

        state = PedestrianState(
            positions=positions,
            velocities=velocities,
            desired_speed=desired_speed,
            target=target,
            group=group,
        )

        def maintain(s: PedestrianState, _dt: float) -> None:
            x = s.positions[:, 0]
            y = s.positions[:, 1]

            in_corner_or_after = x >= l1 - 2.0
            before_corner = ~in_corner_or_after

            s.target[before_corner, 0] = l1 + w / 2.0
            s.target[before_corner, 1] = w / 2.0

            s.target[in_corner_or_after, 0] = l1 + w / 2.0
            s.target[in_corner_or_after, 1] = w + l2 + 5.0

            exited = y > w + l2
            if np.any(exited):
                s.positions[exited, 0] = rng.uniform(margin, 5.0, size=int(exited.sum()))
                s.positions[exited, 1] = rng.uniform(margin, w - margin, size=int(exited.sum()))
                s.velocities[exited] *= 0.0

        return Simulator(
            state=state,
            walls=self._walls(),
            params=self.params,
            maintain=maintain,
        )
@dataclass
class IntersectionScenario:
    """
    Skrzyżowanie dwóch korytarzy pod kątem prostym.
    """
    name: str = "intersection"
    n_pedestrians: int = 80
    params: ModelParameters = field(default_factory=ModelParameters)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        # Taki sam wymiar jak w zakręcie (Szerokość 30, Wysokość 20)
        return (-1.0, 31.0, -1.0, 21.0)

    def _walls(self) -> list[Wall]:
        # Korytarz poziomy (y: 6 do 14) i pionowy (x: 11 do 19)
        return [
            # Lewa strona poziomego korytarza
            Wall(np.array([0.0, 6.0]), np.array([11.0, 6.0])),
            Wall(np.array([0.0, 14.0]), np.array([11.0, 14.0])),
            # Prawa strona poziomego korytarza
            Wall(np.array([19.0, 6.0]), np.array([30.0, 6.0])),
            Wall(np.array([19.0, 14.0]), np.array([30.0, 14.0])),
            # Dolna strona pionowego korytarza
            Wall(np.array([11.0, 0.0]), np.array([11.0, 6.0])),
            Wall(np.array([19.0, 0.0]), np.array([19.0, 6.0])),
            # Górna strona pionowego korytarza
            Wall(np.array([11.0, 14.0]), np.array([11.0, 20.0])),
            Wall(np.array([19.0, 14.0]), np.array([19.0, 20.0])),
        ]

    def build(self, rng: np.random.Generator) -> Simulator:
        n_group = self.n_pedestrians // 2
        
        # Grupa 0 idzie z lewej na prawą
        pos_0 = _scatter_positions(n_group, (0.5, 8.0), (6.5, 13.5), min_distance=0.6, rng=rng)
        # Grupa 1 idzie z dołu do góry
        pos_1 = _scatter_positions(n_group, (11.5, 18.5), (0.5, 5.0), min_distance=0.6, rng=rng)
        
        positions = np.vstack([pos_0, pos_1])
        group = np.zeros(self.n_pedestrians, dtype=int)
        group[n_group:] = 1

        desired_speed = _sample_desired_speeds(self.n_pedestrians, self.params, rng)
        velocities = np.zeros((self.n_pedestrians, 2))
        
        target = np.zeros((self.n_pedestrians, 2))
        # Cele
        target[group == 0, 0] = 35.0  # W prawo
        target[group == 0, 1] = positions[group == 0, 1]
        target[group == 1, 0] = positions[group == 1, 0]
        target[group == 1, 1] = 25.0  # W górę

        state = PedestrianState(
            positions=positions, velocities=velocities, 
            desired_speed=desired_speed, target=target, group=group
        )

        def maintain(s: PedestrianState, _dt: float) -> None:
            # Zawijanie dla Grupy 0 (Poziomej)
            g0 = s.group == 0
            exited_right = g0 & (s.positions[:, 0] > 30.0)
            if np.any(exited_right):
                s.positions[exited_right, 0] = rng.uniform(0.1, 2.0, size=int(exited_right.sum()))
                s.positions[exited_right, 1] = rng.uniform(6.5, 13.5, size=int(exited_right.sum()))
                s.velocities[exited_right] *= 0.0

            # Zawijanie dla Grupy 1 (Pionowej)
            g1 = s.group == 1
            exited_top = g1 & (s.positions[:, 1] > 20.0)
            if np.any(exited_top):
                s.positions[exited_top, 0] = rng.uniform(11.5, 18.5, size=int(exited_top.sum()))
                s.positions[exited_top, 1] = rng.uniform(0.1, 2.0, size=int(exited_top.sum()))
                s.velocities[exited_top] *= 0.0

        return Simulator(state=state, walls=self._walls(), params=self.params, maintain=maintain)


@dataclass
class MetroStationScenario:
    """
    Stacja metra z filarami i odpychającym stacjonarnym agentem.
    """
    name: str = "metro"
    n_pedestrians: int = 60
    params: ModelParameters = field(default_factory=ModelParameters)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (-1.0, 31.0, -1.0, 21.0)

    def _walls(self) -> list[Wall]:
        walls = [
            # Główne ściany korytarza
            Wall(np.array([0.0, 0.0]), np.array([30.0, 0.0])),
            Wall(np.array([0.0, 20.0]), np.array([30.0, 20.0])),
        ]
        
        # Funkcja do tworzenia filarów (rombów/kwadratów)
        def add_pillar(cx: float, cy: float, r: float):
            walls.extend([
                Wall(np.array([cx-r, cy]), np.array([cx, cy+r])),
                Wall(np.array([cx, cy+r]), np.array([cx+r, cy])),
                Wall(np.array([cx+r, cy]), np.array([cx, cy-r])),
                Wall(np.array([cx, cy-r]), np.array([cx-r, cy])),
            ])
            
        # Dwa słupy na drodze
        add_pillar(10.0, 6.0, 1.5)
        add_pillar(10.0, 14.0, 1.5)
        add_pillar(22.0, 10.0, 1.5)
        return walls

    def build(self, rng: np.random.Generator) -> Simulator:
        positions = _scatter_positions(
            self.n_pedestrians, (0.5, 5.0), (1.0, 19.0), min_distance=0.6, rng=rng
        )
        
        # Agent 0 będzie odpychającym obserwatorem na środku
        positions[0] = np.array([16.0, 10.0])
        
        group = np.zeros(self.n_pedestrians, dtype=int)
        group[0] = 1 # Oznaczamy go innym kolorem
        
        desired_speed = _sample_desired_speeds(self.n_pedestrians, self.params, rng)
        desired_speed[0] = 0.0 # Agent 0 nigdzie nie idzie
        
        velocities = np.zeros((self.n_pedestrians, 2))
        target = np.column_stack([np.full(self.n_pedestrians, 35.0), positions[:, 1]])
        target[0] = positions[0]

        state = PedestrianState(
            positions=positions, velocities=velocities,
            desired_speed=desired_speed, target=target, group=group
        )

        def maintain(s: PedestrianState, dt: float) -> None:
            # 1. Zabetonowanie Agenta 0
            s.positions[0] = np.array([16.0, 10.0])
            s.velocities[0] = np.array([0.0, 0.0])
            s.target[0] = s.positions[0]

            # 2. Ręczne odpychanie od Agenta 0 (Asymetryczna strefa prywatna)
            diff = s.positions[1:] - s.positions[0]
            dist = np.linalg.norm(diff, axis=-1)
            
            # Silny gradient wykładniczy wypychający tłum
            force_mag = 40.0 * np.exp(-dist / 2.5) 
            push = (diff / np.maximum(dist[:, None], 1e-12)) * force_mag[:, None]
            
            # Aplikacja pchnięcia tylko do pozostałych agentów
            s.velocities[1:] += push * dt

            # 3. Zawijanie tłumu
            exited = s.positions[:, 0] > 30.0
            exited[0] = False # Oprócz Agenta 0
            if np.any(exited):
                s.positions[exited, 0] = rng.uniform(0.1, 2.0, size=int(exited.sum()))
                s.positions[exited, 1] = rng.uniform(1.0, 19.0, size=int(exited.sum()))
                s.velocities[exited] *= 0.0

        return Simulator(state=state, walls=self._walls(), params=self.params, maintain=maintain)