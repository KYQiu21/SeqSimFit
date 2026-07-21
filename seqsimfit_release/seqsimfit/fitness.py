"""Fitness transformations and raw proxy calculators.

All proxy calculators exposed here follow a single convention: lower proxy
values are better. The sigmoid transformation then maps them to fitness in
``(0, 1)``.
"""

from __future__ import annotations

from typing import Literal
import math

import numpy as np

try:
    from .utils import ll, run_model, score_variants
except ImportError:  # flat-file compatibility
    from utils import ll, run_model, score_variants


def proxy2fitness(energy: float, ref_energy: float, beta: float) -> float:
    """Convert a lower-is-better proxy into a numerically stable sigmoid fitness."""

    x = (float(ref_energy) - float(energy)) * float(beta)
    if x >= 0:
        value = 1.0 / (1.0 + math.exp(-min(x, 745.0)))
    else:
        exp_x = math.exp(max(x, -745.0))
        value = exp_x / (1.0 + exp_x)
    return float(value)


def esmif_likelihood_calculator(
    coords,
    mutant_seq: str,
    model,
    alphabet,
    chain_id: str,
    *,
    score_mode: Literal["legacy_probability", "nll"] = "legacy_probability",
) -> float:
    """Calculate an ESM-IF lower-is-better sequence proxy.

    ``legacy_probability`` exactly preserves the historical project behavior:
    negative sum of assigned residue probabilities. ``nll`` is the conventional
    negative log-likelihood and is recommended for new analyses.
    """

    token_probs = run_model(coords, mutant_seq, model, alphabet, chain_target=chain_id)
    _, probabilities = score_variants(mutant_seq, token_probs, alphabet)
    probabilities = np.asarray(probabilities, dtype=float)
    if score_mode == "legacy_probability":
        return float(-np.nansum(probabilities))
    if score_mode == "nll":
        return float(-np.nansum(np.log(np.clip(probabilities, 1e-12, 1.0))))
    raise ValueError(f"Unsupported ESM-IF score_mode: {score_mode!r}")


def esm2_likelihood_calculator(mutant_seq, model, alphabet, batch_converter, *, device=None) -> float:
    """Pseudo-perplexity proxy: mean masked-token negative log-likelihood."""

    import torch

    model_device = next(model.parameters()).device if device is None else torch.device(device)
    _, _, batch_tokens = batch_converter([("seq", mutant_seq)])
    tokens = batch_tokens.to(model_device)
    log_likelihoods: list[float] = []

    with torch.no_grad():
        for i in range(1, tokens.size(1) - 1):
            masked_tokens = tokens.clone()
            masked_tokens[0, i] = alphabet.mask_idx
            logits = model(masked_tokens)["logits"]
            log_probs = torch.log_softmax(logits[0, i], dim=0)
            log_likelihoods.append(float(log_probs[tokens[0, i]].item()))

    if not log_likelihoods:
        raise ValueError("Cannot score an empty sequence with ESM-2.")
    return float(-np.mean(log_likelihoods))


def progen_likelihood_caculator(mutant_seq, model, tokenizer) -> float:
    """Bidirectional ProGen mean negative log-likelihood.

    The historical misspelling ``caculator`` is retained for compatibility.
    """

    sequence = "1" + mutant_seq + "2"
    ll_lr = ll(tokens=sequence, model=model, tokenizer=tokenizer, reduction="mean")
    ll_rl = ll(tokens=sequence[::-1], model=model, tokenizer=tokenizer, reduction="mean")
    return float(-0.5 * (ll_lr + ll_rl))


# Correctly spelled alias for new code.
progen_likelihood_calculator = progen_likelihood_caculator


def proteinmpnn_cce_calculator(sequence: str, logits_or_log_probs: np.ndarray) -> float:
    """Mean ProteinMPNN negative log-likelihood for ``sequence``.

    The input may be raw logits or already-normalized log probabilities.
    """

    import torch

    alphabet = "ACDEFGHIKLMNPQRSTVWYX"
    aa_to_index = {aa: i for i, aa in enumerate(alphabet)}
    matrix = torch.as_tensor(np.asarray(logits_or_log_probs), dtype=torch.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(sequence):
        raise ValueError(
            f"ProteinMPNN profile shape {tuple(matrix.shape)} is incompatible with sequence length {len(sequence)}."
        )
    labels = []
    for aa in sequence:
        if aa not in aa_to_index:
            raise ValueError(f"Unsupported residue for ProteinMPNN scoring: {aa!r}")
        labels.append(aa_to_index[aa])
    labels_t = torch.tensor(labels, dtype=torch.long)
    log_probs = torch.log_softmax(matrix, dim=-1)
    return float(torch.nn.functional.nll_loss(log_probs, labels_t, reduction="mean").item())


def rosetta_dg_calculator(
    model_pose,
    mutate_resi_idx: int,
    mutate_resi_original_aa: str,
    mutate_resi_aa: str,
    pack_radius: float,
    scorefxn,
    path=None,
):
    """Score a point mutation and return the proposed PyRosetta pose."""

    del path
    from utils import clone_rosetta_pose, initialize_pyrosetta

    initialize_pyrosetta()
    from pyrosetta.toolbox import mutate_residue

    original_pose = clone_rosetta_pose(model_pose)
    mutate_residue(original_pose, mutate_resi_idx, mutate_resi_original_aa, pack_radius=pack_radius)
    energy_original = float(scorefxn(original_pose))

    mutant_pose = clone_rosetta_pose(model_pose)
    mutate_residue(mutant_pose, mutate_resi_idx, mutate_resi_aa, pack_radius=pack_radius)
    energy_mutant = float(scorefxn(mutant_pose))
    return energy_original, energy_mutant, mutant_pose


def plmdca_energy_calculator(sequence: np.ndarray, J: np.ndarray, h: np.ndarray) -> float:
    """Compute a lower-is-better plmDCA statistical energy."""

    sequence = np.asarray(sequence, dtype=int)
    L = len(sequence)
    if h.shape[0] != L or J.shape[0] != L or J.shape[1] != L:
        raise ValueError("DCA parameter dimensions do not match sequence length.")

    states = sequence - 1
    if np.any(states < 0) or np.any(states >= h.shape[1]):
        raise ValueError("Sequence contains a state outside the DCA alphabet.")

    score = float(np.sum(h[np.arange(L), states]))
    for i in range(L):
        score += float(np.sum([J[i, j, states[i], states[j]] for j in range(i + 1, L)]))
    return -score


def multi_plmdca_energy_calculator(alignment_matrix, J_matrix, H_matrix) -> np.ndarray:
    alignment = np.asarray(alignment_matrix)
    if alignment.ndim != 2:
        raise ValueError("alignment_matrix must be two-dimensional (L x N).")
    return np.asarray(
        [plmdca_energy_calculator(alignment[:, i], J_matrix, H_matrix) for i in range(alignment.shape[1])]
    )
