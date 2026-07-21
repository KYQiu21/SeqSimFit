"""Evolutionary acceptance models and mutation proposal utilities.

This module intentionally has no heavy machine-learning or structural-biology
imports, so it can be imported and tested in a lightweight environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence
import math
import random

import numpy as np

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
_EPS = np.finfo(float).tiny


@dataclass(frozen=True)
class Mutation:
    """A single amino-acid substitution.

    ``position`` is zero-based. ``rosetta_position`` provides the corresponding
    one-based index expected by PyRosetta.
    """

    position: int
    original_aa: str
    new_aa: str

    @property
    def rosetta_position(self) -> int:
        return self.position + 1


def _validate_fitness(current_fitness: float, old_fitness: float) -> tuple[float, float]:
    current = float(current_fitness)
    old = float(old_fitness)
    if not (math.isfinite(current) and math.isfinite(old)):
        raise ValueError("Fitness values must be finite.")
    if current < 0 or old < 0:
        raise ValueError("Fixation models require non-negative fitness values.")
    return current, old


def kimura_fixation_prob(current_fitness: float, old_fitness: float, N: int) -> float:
    """Kimura fixation probability with stable neutral handling.

    The formula follows the convention used in the historical project code:
    ``s = current / old - 1`` and denominator ``1 - exp(-4 N s)``.
    At neutrality, the analytical limit is ``1 / (2N)``.
    """

    current, old = _validate_fitness(current_fitness, old_fitness)
    if N <= 0:
        raise ValueError("N must be a positive integer.")
    if old <= _EPS:
        return 1.0 if current > old else 1.0 / (2.0 * N)

    s = current / old - 1.0
    if abs(s) < 1e-12:
        return 1.0 / (2.0 * N)

    numerator = -math.expm1(-2.0 * s)
    denominator = -math.expm1(-4.0 * N * s)
    if denominator == 0.0:
        return 1.0 if s > 0 else 0.0
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def mcmc_fixation_prob(current_fitness: float, old_fitness: float, beta: float) -> float:
    """Metropolis acceptance probability for a quantity where larger is better."""

    current, old = _validate_fitness(current_fitness, old_fitness)
    log_ratio = float(beta) * (current - old)
    if log_ratio >= 0:
        return 1.0
    return float(math.exp(max(log_ratio, -745.0)))


def birthdeath_fixation_prob(current_fitness: float, old_fitness: float, N: int) -> float:
    """Exact birth-death fixation probability used by the legacy implementation."""

    current, old = _validate_fitness(current_fitness, old_fitness)
    if N <= 0:
        raise ValueError("N must be a positive integer.")
    if old <= _EPS:
        return 1.0 if current > old else 1.0 / N

    ratio = current / old
    if abs(ratio - 1.0) < 1e-12:
        return 1.0 / N

    # Evaluate (1-r^2)/(1-r^(2N)) in log space when needed.
    try:
        numerator = 1.0 - ratio**2
        denominator = 1.0 - ratio ** (2 * N)
        value = numerator / denominator
    except OverflowError:
        value = 1.0 if ratio > 1.0 else 0.0
    return float(np.clip(value, 0.0, 1.0))


def accelerated_birthdeath_fixation_prob(
    current_fitness: float,
    old_fitness: float,
    N: int,
) -> float:
    """Accelerated birth-death acceptance used throughout the original project.

    Beneficial and neutral proposals are accepted with probability one. A
    deleterious proposal is accepted with ``(current/old) ** (2N-2)``.
    """

    current, old = _validate_fitness(current_fitness, old_fitness)
    if N <= 0:
        raise ValueError("N must be a positive integer.")
    if current >= old:
        return 1.0
    if current <= 0.0 or old <= _EPS:
        return 0.0

    exponent = max(0, 2 * N - 2)
    log_p = exponent * math.log(current / old)
    return float(math.exp(max(log_p, -745.0)))


def fixation_probability(
    model: str,
    current_fitness: float,
    old_fitness: float,
    *,
    population_size: int,
    mcmc_beta: float = 1.0,
) -> float:
    """Dispatch to a named acceptance/fixation model."""

    key = model.lower().replace("-", "_")
    if key in {"birthdeath", "accelerated_birthdeath", "accelerated"}:
        return accelerated_birthdeath_fixation_prob(current_fitness, old_fitness, population_size)
    if key in {"exact_birthdeath", "birthdeath_exact"}:
        return birthdeath_fixation_prob(current_fitness, old_fitness, population_size)
    if key == "kimura":
        return kimura_fixation_prob(current_fitness, old_fitness, population_size)
    if key in {"mcmc", "metropolis"}:
        return mcmc_fixation_prob(current_fitness, old_fitness, mcmc_beta)
    raise ValueError(f"Unsupported fixation model: {model!r}")


def mutator(
    sequence: str,
    allowed_mutation_dict: Mapping[int, Sequence[str]],
    rng: random.Random | None = None,
) -> tuple[str, int, str, str]:
    """Propose a non-synonymous substitution from a position-specific alphabet.

    This preserves the historical return signature. Positions without any
    alternative amino acid are excluded rather than producing silent no-op moves.
    """

    rng = rng or random
    mutable: list[tuple[int, list[str]]] = []
    for position, current_aa in enumerate(sequence):
        if current_aa == "-":
            continue
        candidates = [
            aa for aa in allowed_mutation_dict.get(position, ())
            if aa in AA_ORDER and aa != current_aa
        ]
        if candidates:
            mutable.append((position, candidates))

    if not mutable:
        raise RuntimeError("No valid non-synonymous mutation is available.")

    mutation_pos, candidates = rng.choice(mutable)
    original_aa = sequence[mutation_pos]
    new_aa = rng.choice(candidates)
    mutated_sequence = sequence[:mutation_pos] + new_aa + sequence[mutation_pos + 1 :]
    return mutated_sequence, mutation_pos, original_aa, new_aa


def lg_mutator(
    sequence: str,
    lg_matrix: np.ndarray | str | Path,
    rng: np.random.Generator | None = None,
) -> tuple[str, int, str, str]:
    """Propose a substitution according to an LG exchangeability matrix.

    Unlike the legacy function, the matrix path is not hard-coded. Pass a loaded
    20x20 array to avoid disk I/O on every proposal.
    """

    matrix = np.load(lg_matrix) if isinstance(lg_matrix, (str, Path)) else np.asarray(lg_matrix)
    if matrix.shape != (20, 20):
        raise ValueError(f"LG matrix must have shape (20, 20), got {matrix.shape}.")

    rng = rng or np.random.default_rng()
    lg_order = "ARNDCQEGHILKMFPSTWYV"
    aa_to_index = {aa: i for i, aa in enumerate(lg_order)}

    mutable_positions = [i for i, aa in enumerate(sequence) if aa in aa_to_index]
    if not mutable_positions:
        raise RuntimeError("No standard amino-acid position is available for LG mutation.")

    mutation_pos = int(rng.choice(mutable_positions))
    original_aa = sequence[mutation_pos]
    idx = aa_to_index[original_aa]
    rates = np.asarray(matrix[idx], dtype=float).copy()
    rates[idx] = 0.0
    rates[~np.isfinite(rates)] = 0.0
    rates[rates < 0.0] = 0.0
    total = rates.sum()
    if total <= 0.0:
        raise ValueError(f"LG row for {original_aa} has no positive off-diagonal rate.")

    new_idx = int(rng.choice(20, p=rates / total))
    new_aa = lg_order[new_idx]
    mutated_sequence = sequence[:mutation_pos] + new_aa + sequence[mutation_pos + 1 :]
    return mutated_sequence, mutation_pos, original_aa, new_aa
