from __future__ import annotations

import numpy as np

from .model import ModelParameters, PedestrianState, Wall

_EPS = 1e-12


def _safe_unit(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Zwraca normę i wektor jednostkowy wzdłuż ostatniej osi."""
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    unit = v / np.maximum(norm, _EPS)
    return norm.squeeze(-1), unit


def desired_direction(state: PedestrianState) -> np.ndarray:
    """Jednostkowy wektor w kierunku punktu docelowego, równanie (1)."""
    _, e = _safe_unit(state.target - state.positions)
    return e


def acceleration_force(
    state: PedestrianState, params: ModelParameters
) -> np.ndarray:
    """Siła dążenia do preferowanej prędkości, równanie (2).

    F^0 = (v0 * e - v) / tau
    """
    e = desired_direction(state)
    desired_velocity = state.desired_speed[:, None] * e
    return (desired_velocity - state.velocities) / params.tau


def _fov_weight(
    e_alpha: np.ndarray, force: np.ndarray, params: ModelParameters
) -> np.ndarray:
    """Waga pola widzenia, równanie (7).
    """
    _, force_hat = _safe_unit(force)
    cos_angle = -(e_alpha * force_hat).sum(axis=-1)
    in_fov = cos_angle >= np.cos(params.phi)
    return np.where(in_fov, 1.0, params.c)


def pedestrian_repulsion(
    state: PedestrianState, params: ModelParameters
) -> np.ndarray:
    """Odpychanie między pieszymi, równania (3)-(4).
    """
    n = state.n
    if n < 2:
        return np.zeros((n, 2))

    # Parami przesunięcia r_ab = r_alpha - r_beta, kształt (N, N, 2).
    r_ab = state.positions[:, None, :] - state.positions[None, :, :]
    norm_r_ab = np.linalg.norm(r_ab, axis=-1)

    # Przewidywany krok pieszego beta.
    step_b = params.delta_t * state.velocities
    step_b_len = np.linalg.norm(step_b, axis=-1)

    r_ab_shift = r_ab - step_b[None, :, :]
    norm_r_ab_shift = np.linalg.norm(r_ab_shift, axis=-1)

    sum_norms = norm_r_ab + norm_r_ab_shift
    two_b = np.sqrt(np.maximum(sum_norms**2 - step_b_len[None, :] ** 2, 0.0))
    b = two_b / 2.0

    r_ab_hat = r_ab / np.maximum(norm_r_ab[..., None], _EPS)
    r_ab_shift_hat = r_ab_shift / np.maximum(norm_r_ab_shift[..., None], _EPS)

    grad_b = sum_norms[..., None] * (r_ab_hat + r_ab_shift_hat) / np.maximum(
        4.0 * b[..., None], _EPS
    )
    magnitude = (params.V0 / params.sigma) * np.exp(-b / params.sigma)
    force_pairs = magnitude[..., None] * grad_b

    e_alpha = desired_direction(state)
    weight = _fov_weight(e_alpha[:, None, :], force_pairs, params)
    force_pairs = force_pairs * weight[..., None]

    # Zerujemy oddziaływanie pieszego z samym sobą.
    diag = np.arange(n)
    force_pairs[diag, diag] = 0.0

    return force_pairs.sum(axis=1)


def wall_repulsion(
    state: PedestrianState,
    walls: list[Wall],
    params: ModelParameters,
) -> np.ndarray:
    """
    Odpychanie od ścian, równanie (5).
    """
    total = np.zeros_like(state.positions)
    if not walls:
        return total

    e_alpha = desired_direction(state)
    for wall in walls:
        nearest = wall.nearest_point(state.positions)
        r = state.positions - nearest
        dist, r_hat = _safe_unit(r)
        magnitude = (params.U0 / params.R) * np.exp(-dist / params.R)
        force = magnitude[:, None] * r_hat
        weight = _fov_weight(e_alpha, force, params)
        total = total + force * weight[:, None]

    return total


def total_force(
    state: PedestrianState,
    walls: list[Wall],
    params: ModelParameters,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Suma wszystkich sił, równanie (9), opcjonalnie szum"""
    f = acceleration_force(state, params)
    f = f + pedestrian_repulsion(state, params)
    f = f + wall_repulsion(state, walls, params)
    if params.noise_std > 0.0:
        generator = rng if rng is not None else np.random.default_rng()
        f = f + generator.normal(0.0, params.noise_std, size=f.shape)
    return f
