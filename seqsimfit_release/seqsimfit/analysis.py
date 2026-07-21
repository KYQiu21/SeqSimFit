"""Trajectory analysis utilities for SeqSimFit.

The functions in this module are deliberately independent of any scoring
backend.  They operate on accepted sequence trajectories and can therefore be
used with ProteinMPNN-, ProGen-, ESM-, DCA-, Rosetta-, callable-, or composite-
objective simulations.

The core analyses mirror the trajectory-level quantities used in the SeqSimFit
manuscript:

* sequence divergence from the starting sequence;
* site-specific substitution counts;
* site-wise amino-acid distributions and KL divergence to a reference MSA;
* empirical substitution count, rate, and exchangeability matrices.

Only NumPy is required.  Plotting is intentionally kept out of the core module
so that notebooks can choose their own presentation style.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence, Any
import math

import numpy as np

try:
    from .model import Mutation
except ImportError:  # flat-file compatibility
    from model import Mutation


AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
_GAP_CHARS = frozenset("-.")


@dataclass(frozen=True)
class MutationEvent:
    """One accepted sequence transition.

    ``step`` is the one-based accepted-substitution index and ``position`` is
    zero-based.  A trajectory can contain more than one event per transition
    when ``require_single_substitution=False`` is used.
    """

    step: int
    position: int
    original_aa: str
    new_aa: str

    @property
    def position_one_based(self) -> int:
        return self.position + 1

    @property
    def label(self) -> str:
        return f"{self.original_aa}{self.position_one_based}{self.new_aa}"


@dataclass(frozen=True)
class PositionFrequencyProfile:
    """Position-specific amino-acid counts and frequencies."""

    alphabet: str
    counts: np.ndarray
    frequencies: np.ndarray
    occupancy: np.ndarray
    n_sequences: int


@dataclass(frozen=True)
class KLDivergenceResult:
    """Site-wise and aggregate KL divergence to a reference alignment."""

    sitewise: np.ndarray
    weights: np.ndarray
    aggregate: float
    normalization: str
    reference_profile: PositionFrequencyProfile = field(repr=False)
    simulated_profile: PositionFrequencyProfile = field(repr=False)


@dataclass(frozen=True)
class SubstitutionModel:
    """Empirical continuous-time substitution model estimated from trajectories."""

    alphabet: str
    counts: np.ndarray
    exposures: np.ndarray
    equilibrium_frequencies: np.ndarray
    rate_matrix: np.ndarray
    exchangeability: np.ndarray
    total_substitutions: int
    transitions_checked: int
    single_substitution_fraction: float


@dataclass(frozen=True)
class CorrelationResult:
    method: str
    coefficient: float
    n_valid: int


@dataclass(frozen=True)
class TrajectoryAnalysis:
    """A compact analysis-ready representation of one accepted trajectory."""

    sequences: tuple[str, ...]
    steps: np.ndarray
    mutations: tuple[MutationEvent, ...]
    hamming_to_start: np.ndarray
    identity_to_start: np.ndarray
    site_substitution_counts: np.ndarray
    fitness: np.ndarray | None = None
    proxies: dict[str, np.ndarray] = field(default_factory=dict)
    component_fitness: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def accepted_steps(self) -> int:
        return max(0, len(self.sequences) - 1)

    @property
    def sequence_length(self) -> int:
        return len(self.sequences[0])

    @property
    def final_identity(self) -> float:
        return float(self.identity_to_start[-1])


def _coerce_sequences(sequences: Sequence[str] | Iterable[str], *, name: str = "sequences") -> tuple[str, ...]:
    seqs = tuple(str(seq).upper() for seq in sequences)
    if not seqs:
        raise ValueError(f"{name} must contain at least one sequence.")
    length = len(seqs[0])
    if length == 0:
        raise ValueError(f"{name} cannot contain empty sequences.")
    for index, seq in enumerate(seqs):
        if len(seq) != length:
            raise ValueError(
                f"All entries in {name} must have the same length; "
                f"entry 0 has length {length}, entry {index} has length {len(seq)}."
            )
    return seqs


def slice_trajectory(
    sequences: Sequence[str],
    start_step: int | None = None,
    end_step: int | None = None,
) -> tuple[str, ...]:
    """Slice a trajectory using ordinary Python half-open indexing."""

    seqs = _coerce_sequences(sequences, name="trajectory")
    sliced = seqs[slice(start_step, end_step)]
    if not sliced:
        raise ValueError("The requested trajectory slice is empty.")
    return sliced


def remove_consecutive_duplicates(sequences: Sequence[str]) -> tuple[str, ...]:
    """Remove consecutive duplicate sequence states while preserving order."""

    seqs = _coerce_sequences(sequences)
    result = [seqs[0]]
    result.extend(seq for seq, previous in zip(seqs[1:], seqs[:-1]) if seq != previous)
    return tuple(result)


def hamming_distance(sequence_a: str, sequence_b: str, *, ignore_gaps: bool = False) -> int:
    """Return the Hamming distance between two equal-length sequences."""

    a, b = str(sequence_a).upper(), str(sequence_b).upper()
    if len(a) != len(b):
        raise ValueError("Sequences must have the same length.")
    if ignore_gaps:
        return sum(x != y for x, y in zip(a, b) if x not in _GAP_CHARS and y not in _GAP_CHARS)
    return sum(x != y for x, y in zip(a, b))


def sequence_identity(sequence_a: str, sequence_b: str, *, ignore_gaps: bool = False) -> float:
    """Return fractional sequence identity in the interval [0, 1]."""

    a, b = str(sequence_a).upper(), str(sequence_b).upper()
    if len(a) != len(b):
        raise ValueError("Sequences must have the same length.")
    if ignore_gaps:
        comparable = [(x, y) for x, y in zip(a, b) if x not in _GAP_CHARS and y not in _GAP_CHARS]
        if not comparable:
            return float("nan")
        return sum(x == y for x, y in comparable) / len(comparable)
    if not a:
        return float("nan")
    return sum(x == y for x, y in zip(a, b)) / len(a)


def identity_trajectory(sequences: Sequence[str], reference: str | None = None) -> np.ndarray:
    """Sequence identity of each trajectory state to a reference sequence."""

    seqs = _coerce_sequences(sequences, name="trajectory")
    reference = seqs[0] if reference is None else str(reference).upper()
    if len(reference) != len(seqs[0]):
        raise ValueError("Reference length differs from trajectory sequence length.")
    return np.asarray([sequence_identity(reference, seq) for seq in seqs], dtype=float)


def hamming_trajectory(sequences: Sequence[str], reference: str | None = None) -> np.ndarray:
    """Hamming distance of each trajectory state to a reference sequence."""

    seqs = _coerce_sequences(sequences, name="trajectory")
    reference = seqs[0] if reference is None else str(reference).upper()
    if len(reference) != len(seqs[0]):
        raise ValueError("Reference length differs from trajectory sequence length.")
    return np.asarray([hamming_distance(reference, seq) for seq in seqs], dtype=int)


def mutation_events(
    sequences: Sequence[str],
    *,
    require_single_substitution: bool = True,
) -> tuple[MutationEvent, ...]:
    """Extract accepted mutation events from adjacent trajectory states."""

    seqs = _coerce_sequences(sequences, name="trajectory")
    events: list[MutationEvent] = []
    for step, (before, after) in enumerate(zip(seqs[:-1], seqs[1:]), start=1):
        changed = [i for i, (a, b) in enumerate(zip(before, after)) if a != b]
        if require_single_substitution and len(changed) != 1:
            raise ValueError(
                f"Trajectory transition {step} contains {len(changed)} substitutions; "
                "expected exactly one."
            )
        for position in changed:
            events.append(
                MutationEvent(
                    step=step,
                    position=position,
                    original_aa=before[position],
                    new_aa=after[position],
                )
            )
    return tuple(events)


def site_substitution_counts(
    sequences: Sequence[str],
    *,
    require_single_substitution: bool = True,
) -> np.ndarray:
    """Count accepted substitution events at every sequence position.

    This is the site-rate proxy used in the manuscript: repeated reversions and
    parallel substitutions at the same site are counted as separate events.
    """

    seqs = _coerce_sequences(sequences, name="trajectory")
    counts = np.zeros(len(seqs[0]), dtype=int)
    for event in mutation_events(seqs, require_single_substitution=require_single_substitution):
        counts[event.position] += 1
    return counts


def site_residue_diversity(sequences: Sequence[str], *, subtract_one: bool = True) -> np.ndarray:
    """Number of distinct residue states visited at each site.

    This quantity appeared in exploratory notebooks but is not a substitution
    rate: revisits and reversions are intentionally ignored.  It is therefore
    named explicitly as diversity rather than evolutionary rate.
    """

    seqs = _coerce_sequences(sequences, name="trajectory")
    array = np.asarray([list(seq) for seq in seqs])
    values = np.asarray([len(set(array[:, i])) for i in range(array.shape[1])], dtype=int)
    return np.maximum(values - 1, 0) if subtract_one else values


def aggregate_site_substitution_counts(
    trajectories: Sequence[Sequence[str]],
    *,
    reduction: str = "mean",
    require_single_substitution: bool = True,
) -> np.ndarray:
    """Aggregate site substitution counts across replicate trajectories."""

    if not trajectories:
        raise ValueError("At least one trajectory is required.")
    matrix = np.vstack(
        [
            site_substitution_counts(
                trajectory,
                require_single_substitution=require_single_substitution,
            )
            for trajectory in trajectories
        ]
    )
    reduction = reduction.lower()
    if reduction == "none":
        return matrix
    if reduction == "mean":
        return matrix.mean(axis=0)
    if reduction == "sum":
        return matrix.sum(axis=0)
    if reduction == "median":
        return np.median(matrix, axis=0)
    raise ValueError("reduction must be one of: 'none', 'mean', 'sum', 'median'.")


def position_frequencies(
    sequences: Sequence[str],
    *,
    alphabet: str = AA_ALPHABET,
    pseudocount: float = 0.0,
) -> PositionFrequencyProfile:
    """Compute amino-acid frequencies independently at each alignment column.

    Gaps and non-alphabet symbols are excluded from counts. ``occupancy`` is the
    fraction of input sequences containing a recognized amino acid at each site.
    """

    seqs = _coerce_sequences(sequences, name="alignment")
    alphabet = str(alphabet)
    if len(set(alphabet)) != len(alphabet):
        raise ValueError("alphabet must contain unique symbols.")
    if pseudocount < 0:
        raise ValueError("pseudocount must be non-negative.")

    index = {aa: i for i, aa in enumerate(alphabet)}
    length = len(seqs[0])
    counts = np.zeros((length, len(alphabet)), dtype=float)
    valid_counts = np.zeros(length, dtype=float)
    for seq in seqs:
        for position, aa in enumerate(seq):
            if aa in index:
                counts[position, index[aa]] += 1.0
                valid_counts[position] += 1.0

    smoothed = counts + float(pseudocount)
    totals = smoothed.sum(axis=1, keepdims=True)
    frequencies = np.divide(
        smoothed,
        totals,
        out=np.zeros_like(smoothed),
        where=totals > 0,
    )
    occupancy = valid_counts / len(seqs)
    return PositionFrequencyProfile(
        alphabet=alphabet,
        counts=counts,
        frequencies=frequencies,
        occupancy=occupancy,
        n_sequences=len(seqs),
    )


def amino_acid_composition(
    sequences: Sequence[str],
    *,
    alphabet: str = AA_ALPHABET,
    pseudocount: float = 0.0,
) -> np.ndarray:
    """Global amino-acid composition pooled across sequences and sites."""

    profile = position_frequencies(sequences, alphabet=alphabet, pseudocount=0.0)
    counts = profile.counts.sum(axis=0) + float(pseudocount)
    total = counts.sum()
    return counts / total if total > 0 else np.zeros_like(counts)


def alignment_kl_divergence(
    reference_sequences: Sequence[str],
    simulated_sequences: Sequence[str],
    *,
    alphabet: str = AA_ALPHABET,
    pseudocount: float = 1e-8,
    normalization: str = "mean",
) -> KLDivergenceResult:
    """Compare simulated site distributions with a reference alignment.

    The direction is explicitly ``D_KL(reference || simulated)``.  Site weights
    are the non-gap occupancy of the reference alignment, matching the weighting
    described in the manuscript.

    ``normalization`` controls the aggregate:

    * ``'mean'``: divide the weighted sum by the sum of valid site weights;
    * ``'sum'``: report the unnormalized weighted sum;
    * ``'length'``: divide the weighted sum by sequence length.
    """

    if pseudocount <= 0:
        raise ValueError("pseudocount must be positive for finite KL divergence.")
    reference = _coerce_sequences(reference_sequences, name="reference alignment")
    simulated = _coerce_sequences(simulated_sequences, name="simulated sequences")
    if len(reference[0]) != len(simulated[0]):
        raise ValueError("Reference and simulated sequences must have the same aligned length.")

    ref_profile = position_frequencies(reference, alphabet=alphabet, pseudocount=pseudocount)
    sim_profile = position_frequencies(simulated, alphabet=alphabet, pseudocount=pseudocount)
    p = ref_profile.frequencies
    q = sim_profile.frequencies
    sitewise = np.sum(np.where(p > 0, p * np.log(p / q), 0.0), axis=1)
    weights = ref_profile.occupancy.copy()
    weighted_sum = float(np.sum(weights * sitewise))

    normalization = normalization.lower()
    if normalization == "sum":
        aggregate = weighted_sum
    elif normalization == "mean":
        denominator = float(weights.sum())
        aggregate = weighted_sum / denominator if denominator > 0 else float("nan")
    elif normalization == "length":
        aggregate = weighted_sum / len(weights)
    else:
        raise ValueError("normalization must be one of: 'mean', 'sum', 'length'.")

    return KLDivergenceResult(
        sitewise=sitewise,
        weights=weights,
        aggregate=float(aggregate),
        normalization=normalization,
        reference_profile=ref_profile,
        simulated_profile=sim_profile,
    )


def _coerce_trajectories(trajectories: Sequence[Sequence[str]] | Sequence[str]) -> tuple[tuple[str, ...], ...]:
    if not trajectories:
        raise ValueError("At least one trajectory is required.")
    first = trajectories[0]  # type: ignore[index]
    if isinstance(first, str):
        return (_coerce_sequences(trajectories, name="trajectory"),)  # type: ignore[arg-type]
    result = tuple(_coerce_sequences(traj, name="trajectory") for traj in trajectories)  # type: ignore[arg-type]
    lengths = {len(traj[0]) for traj in result}
    if len(lengths) != 1:
        raise ValueError("All trajectories must use the same sequence length.")
    return result


def estimate_substitution_model(
    trajectories: Sequence[Sequence[str]] | Sequence[str],
    *,
    alphabet: str = AA_ALPHABET,
    require_single_substitution: bool = True,
    pseudocount: float = 0.0,
) -> SubstitutionModel:
    """Estimate a global rate matrix and symmetric exchangeabilities.

    For each amino acid ``i``, off-diagonal rates are estimated as
    ``q_ij = N_ij / S_i``, where ``N_ij`` is the number of observed ``i -> j``
    events and ``S_i`` is the total exposure of state ``i`` across pre-mutation
    trajectory states.  The matrix is scaled to unit expected substitution rate.
    """

    if pseudocount < 0:
        raise ValueError("pseudocount must be non-negative.")
    trajs = _coerce_trajectories(trajectories)
    alphabet = str(alphabet)
    index = {aa: i for i, aa in enumerate(alphabet)}
    k = len(alphabet)
    counts = np.zeros((k, k), dtype=float)
    exposures = np.zeros(k, dtype=float)
    transition_mutation_counts: list[int] = []

    for trajectory in trajs:
        for step, (before, after) in enumerate(zip(trajectory[:-1], trajectory[1:]), start=1):
            for aa in before:
                if aa in index:
                    exposures[index[aa]] += 1.0
            changed = 0
            for aa_before, aa_after in zip(before, after):
                if aa_before != aa_after and aa_before in index and aa_after in index:
                    counts[index[aa_before], index[aa_after]] += 1.0
                    changed += 1
            transition_mutation_counts.append(changed)
            if require_single_substitution and changed != 1:
                raise ValueError(
                    f"A trajectory transition contains {changed} substitutions; expected exactly one."
                )

    if pseudocount:
        off_diagonal = ~np.eye(k, dtype=bool)
        counts[off_diagonal] += pseudocount

    if exposures.sum() <= 0:
        raise ValueError("No valid amino-acid exposures were observed.")

    equilibrium = exposures / exposures.sum()
    q = np.zeros((k, k), dtype=float)
    for i in range(k):
        if exposures[i] > 0:
            q[i, :] = counts[i, :] / exposures[i]
            q[i, i] = 0.0
            q[i, i] = -q[i, :].sum()

    expected_rate = float(-np.sum(equilibrium * np.diag(q)))
    if expected_rate <= 0:
        raise ValueError("No valid substitutions were observed; the rate matrix cannot be normalized.")
    q /= expected_rate

    exchangeability = np.zeros((k, k), dtype=float)
    for i in range(k):
        for j in range(i + 1, k):
            estimates = []
            if equilibrium[j] > 0:
                estimates.append(q[i, j] / equilibrium[j])
            if equilibrium[i] > 0:
                estimates.append(q[j, i] / equilibrium[i])
            if estimates:
                value = float(np.mean(estimates))
                exchangeability[i, j] = exchangeability[j, i] = value

    checked = len(transition_mutation_counts)
    single_fraction = (
        float(np.mean(np.asarray(transition_mutation_counts) == 1)) if checked else float("nan")
    )
    return SubstitutionModel(
        alphabet=alphabet,
        counts=counts,
        exposures=exposures,
        equilibrium_frequencies=equilibrium,
        rate_matrix=q,
        exchangeability=exchangeability,
        total_substitutions=int(np.rint(counts[~np.eye(k, dtype=bool)].sum() - pseudocount * k * (k - 1))),
        transitions_checked=checked,
        single_substitution_fraction=single_fraction,
    )


def upper_triangle_values(matrix: np.ndarray, *, include_diagonal: bool = False) -> np.ndarray:
    """Return finite upper-triangle entries from a square matrix."""

    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square.")
    offset = 0 if include_diagonal else 1
    values = matrix[np.triu_indices(matrix.shape[0], k=offset)]
    return values[np.isfinite(values)]


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks with tie handling, implemented without SciPy."""

    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def correlate_vectors(
    values_a: Sequence[float],
    values_b: Sequence[float],
    *,
    method: str = "spearman",
) -> CorrelationResult:
    """Pearson or Spearman correlation with pairwise finite-value filtering."""

    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("Input vectors must have the same shape.")
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    if len(a) < 2 or np.all(a == a[0]) or np.all(b == b[0]):
        return CorrelationResult(method=method.lower(), coefficient=float("nan"), n_valid=len(a))

    method = method.lower()
    if method == "spearman":
        a, b = _rankdata(a), _rankdata(b)
    elif method != "pearson":
        raise ValueError("method must be 'spearman' or 'pearson'.")
    coefficient = float(np.corrcoef(a, b)[0, 1])
    return CorrelationResult(method=method, coefficient=coefficient, n_valid=len(a))


def correlate_site_rates(
    simulated_rates: Sequence[float],
    reference_rates: Sequence[float],
    *,
    method: str = "spearman",
) -> CorrelationResult:
    """Compare a simulated site-rate profile with an external reference."""

    return correlate_vectors(simulated_rates, reference_rates, method=method)


def compare_exchangeability(
    estimated: SubstitutionModel | np.ndarray,
    reference: SubstitutionModel | np.ndarray,
    *,
    method: str = "spearman",
    exclude_joint_zeros: bool = True,
) -> CorrelationResult:
    """Compare symmetric exchangeability matrices over 190 amino-acid pairs."""

    a = estimated.exchangeability if isinstance(estimated, SubstitutionModel) else np.asarray(estimated, dtype=float)
    b = reference.exchangeability if isinstance(reference, SubstitutionModel) else np.asarray(reference, dtype=float)
    if a.shape != b.shape or a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("Exchangeability matrices must be square and have matching shapes.")
    indices = np.triu_indices(a.shape[0], k=1)
    va, vb = a[indices], b[indices]
    if exclude_joint_zeros:
        keep = ~((va == 0) & (vb == 0))
        va, vb = va[keep], vb[keep]
    return correlate_vectors(va, vb, method=method)


def analyze_trajectory(
    sequences: Sequence[str],
    *,
    fitness: Sequence[float] | None = None,
    proxies: Mapping[str, Sequence[float]] | None = None,
    component_fitness: Mapping[str, Sequence[float]] | None = None,
    require_single_substitution: bool = True,
) -> TrajectoryAnalysis:
    """Build a reusable trajectory-analysis object from sequence states."""

    seqs = _coerce_sequences(sequences, name="trajectory")
    n_states = len(seqs)

    def optional_array(values: Sequence[float] | None, label: str) -> np.ndarray | None:
        if values is None:
            return None
        array = np.asarray(values, dtype=float)
        if array.shape != (n_states,):
            raise ValueError(f"{label} must contain one value per trajectory state.")
        return array

    def mapping_arrays(values: Mapping[str, Sequence[float]] | None, label: str) -> dict[str, np.ndarray]:
        result: dict[str, np.ndarray] = {}
        for name, series in (values or {}).items():
            array = np.asarray(series, dtype=float)
            if array.shape != (n_states,):
                raise ValueError(f"{label}[{name!r}] must contain one value per trajectory state.")
            result[str(name)] = array
        return result

    return TrajectoryAnalysis(
        sequences=seqs,
        steps=np.arange(n_states, dtype=int),
        mutations=mutation_events(seqs, require_single_substitution=require_single_substitution),
        hamming_to_start=hamming_trajectory(seqs),
        identity_to_start=identity_trajectory(seqs),
        site_substitution_counts=site_substitution_counts(
            seqs,
            require_single_substitution=require_single_substitution,
        ),
        fitness=optional_array(fitness, "fitness"),
        proxies=mapping_arrays(proxies, "proxies"),
        component_fitness=mapping_arrays(component_fitness, "component_fitness"),
    )


def analyze_simulator(simulator: Any) -> TrajectoryAnalysis:
    """Analyze the complete accepted history stored by a SequenceSimulator."""

    sequences = tuple(simulator.accepted_seq_list)
    proxy_history = list(simulator.accepted_proxy_list)
    component_history = list(simulator.accepted_component_fitness_list)
    proxy_names = sorted({name for record in proxy_history for name in record})
    component_names = sorted({name for record in component_history for name in record})
    proxies = {
        name: [record.get(name, float("nan")) for record in proxy_history]
        for name in proxy_names
    }
    components = {
        name: [record.get(name, float("nan")) for record in component_history]
        for name in component_names
    }
    return analyze_trajectory(
        sequences,
        fitness=simulator.accepted_fitness_list,
        proxies=proxies,
        component_fitness=components,
        require_single_substitution=True,
    )


def first_state_at_identity(
    sequences: Sequence[str],
    target_identity: float,
    *,
    reference: str | None = None,
    mode: str = "at_or_below",
) -> tuple[int, str, float]:
    """Select a trajectory state at a requested identity to a reference.

    ``mode='at_or_below'`` returns the first state whose identity is no greater
    than the target. ``mode='closest'`` returns the globally closest state.
    """

    if not 0 <= target_identity <= 1:
        raise ValueError("target_identity must lie in [0, 1].")
    seqs = _coerce_sequences(sequences, name="trajectory")
    identities = identity_trajectory(seqs, reference)
    mode = mode.lower()
    if mode == "at_or_below":
        candidates = np.flatnonzero(identities <= target_identity)
        index = int(candidates[0]) if len(candidates) else int(np.argmin(identities))
    elif mode == "closest":
        index = int(np.argmin(np.abs(identities - target_identity)))
    else:
        raise ValueError("mode must be 'at_or_below' or 'closest'.")
    return index, seqs[index], float(identities[index])


def read_fasta(path: str | Path) -> tuple[str, ...]:
    """Read FASTA sequences without requiring Biopython."""

    sequences: list[str] = []
    current: list[str] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current:
                    sequences.append("".join(current).upper())
                    current = []
            else:
                current.append(line.replace(" ", ""))
        if current:
            sequences.append("".join(current).upper())
    return _coerce_sequences(sequences, name="FASTA alignment")


def write_fasta(
    sequences: Sequence[str],
    path: str | Path,
    *,
    headers: Sequence[str] | None = None,
) -> None:
    """Write equal-length or unaligned sequences to FASTA."""

    seqs = tuple(str(seq).upper() for seq in sequences)
    if not seqs:
        raise ValueError("At least one sequence is required.")
    if headers is None:
        headers = [f"sequence_{i + 1}" for i in range(len(seqs))]
    if len(headers) != len(seqs):
        raise ValueError("headers and sequences must have the same length.")
    with Path(path).open("w", encoding="utf-8") as handle:
        for header, sequence in zip(headers, seqs):
            clean_header = str(header).lstrip(">")
            handle.write(f">{clean_header}\n{sequence}\n")


def trim_alignment_to_query(
    alignment: Sequence[str],
    *,
    query_index: int = 0,
    target_sequence: str | None = None,
) -> tuple[str, ...]:
    """Remove query-gap columns and optionally crop to a target subsequence.

    This is a pure replacement for the exploratory notebook helper that relied
    on a global ``consurf_seq`` variable.
    """

    seqs = _coerce_sequences(alignment, name="alignment")
    if not 0 <= query_index < len(seqs):
        raise IndexError("query_index is outside the alignment.")
    query = seqs[query_index]
    keep = [i for i, aa in enumerate(query) if aa not in _GAP_CHARS]
    trimmed = tuple("".join(seq[i] for i in keep) for seq in seqs)
    if target_sequence is None:
        return trimmed
    target = str(target_sequence).upper().replace("-", "").replace(".", "")
    start = trimmed[query_index].find(target)
    if start < 0:
        raise ValueError("target_sequence was not found in the ungapped query sequence.")
    end = start + len(target)
    return tuple(seq[start:end] for seq in trimmed)
