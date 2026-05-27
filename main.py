"""Punkt wejscia do uruchamiania symulacji modelu sil socjalnych.

Uzycie:
    python main.py --scenario lanes
    python main.py --scenario bottleneck --save bottleneck.mp4
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from social_force import BottleneckScenario, LaneFormationScenario, Scenario
from visualization import AnimationConfig, PedestrianAnimator, save_animation


def build_scenario(name: str, n_pedestrians: int | None) -> Scenario:
    """Tworzy scenariusz o podanej nazwie."""
    if name == "lanes":
        if n_pedestrians is None:
            return LaneFormationScenario()
        return LaneFormationScenario(n_pedestrians=n_pedestrians)
    if name == "bottleneck":
        if n_pedestrians is None:
            return BottleneckScenario()
        return BottleneckScenario(n_per_group=max(1, n_pedestrians // 2))
    raise ValueError(f"Nieznany scenariusz: {name!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        choices=["lanes", "bottleneck"],
        default="lanes",
        help="Wybor scenariusza symulacji.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Liczba pieszych (na grupe w scenariuszu bottleneck).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Ziarno generatora losowego."
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=None,
        help="Maksymalny czas symulacji [s]. Domyslnie 30 gdy uzyto --save.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Plik wyjsciowy (.mp4 / .gif). Gdy pominiete, animacja jest pokazywana.",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    scenario = build_scenario(args.scenario, args.n)
    simulator = scenario.build(rng)

    max_seconds = args.seconds
    if max_seconds is None and args.save:
        max_seconds = 30.0

    animator = PedestrianAnimator(
        simulator=simulator,
        scenario=scenario,
        config=AnimationConfig(),
        max_seconds=max_seconds,
    )
    anim = animator.animate()

    if args.save:
        save_animation(anim, args.save)
        print(f"Zapisano do {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
