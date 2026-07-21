"""Shared utilities for sequence scoring backends.

Heavy optional dependencies are imported lazily. Importing this module does not
initialize PyRosetta, load a neural network, change the working directory, or
require a GPU.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator, Sequence
import sys

import numpy as np

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_WITH_GAP = AA_ORDER + "-"
AMINO_ACID_MAPPING = {aa: i + 1 for i, aa in enumerate(AA_WITH_GAP)}
AMINO_ACID_MAPPING["X"] = 21


def validate_sequence(sequence: str, *, allow_gaps: bool = True) -> str:
    """Normalize and validate an amino-acid sequence."""

    if not isinstance(sequence, str) or not sequence:
        raise ValueError("sequence must be a non-empty string")
    sequence = sequence.strip().upper()
    alphabet = set(AA_ORDER + ("-" if allow_gaps else ""))
    invalid = sorted(set(sequence) - alphabet)
    if invalid:
        raise ValueError(f"Unsupported residues in sequence: {invalid}")
    return sequence


def vec_to_seq(mapping: Iterable[int]) -> str:
    """Convert one-based DCA states to an amino-acid string."""

    chars = []
    for value in mapping:
        index = int(value) - 1
        if index < 0 or index >= len(AA_WITH_GAP):
            raise ValueError(f"Invalid DCA state: {value}")
        chars.append(AA_WITH_GAP[index])
    return "".join(chars)


def seq_to_vec(sequence: str) -> np.ndarray:
    """Convert an amino-acid string to one-based DCA states."""

    try:
        return np.asarray([AMINO_ACID_MAPPING[aa.upper()] for aa in sequence], dtype=np.int16)
    except KeyError as exc:
        raise ValueError(f"Unsupported amino acid for DCA: {exc.args[0]!r}") from exc


def read_fasta_alignment(filename: str | Path, max_gap_fraction: float = 1.0) -> np.ndarray:
    """Read an aligned FASTA file into an ``L x N`` one-based integer matrix.

    Dot characters are treated as alignment placeholders and removed consistently
    from every sequence. This fixes the length assertion in the historical code,
    which failed whenever dots were actually present.
    """

    from Bio import SeqIO

    if not 0.0 <= max_gap_fraction <= 1.0:
        raise ValueError("max_gap_fraction must be between 0 and 1")

    raw_sequences = [str(record.seq).upper().replace("X", "-") for record in SeqIO.parse(str(filename), "fasta")]
    if not raw_sequences:
        raise ValueError(f"No sequences found in {filename}")

    raw_length = len(raw_sequences[0])
    if any(len(seq) != raw_length for seq in raw_sequences):
        raise ValueError("Inputs are not aligned: FASTA sequences have different lengths.")

    valid_indices = [i for i, char in enumerate(raw_sequences[0]) if char != "."]
    sequences: list[str] = []
    for seq in raw_sequences:
        if any((seq[i] == ".") != (raw_sequences[0][i] == ".") for i in range(raw_length)):
            raise ValueError("Dot placeholders are inconsistent across aligned sequences.")
        cleaned = "".join(seq[i] for i in valid_indices)
        invalid = set(cleaned) - set(AA_WITH_GAP)
        if invalid:
            raise ValueError(f"Unsupported alignment symbols: {sorted(invalid)}")
        if cleaned.count("-") / max(1, len(cleaned)) <= max_gap_fraction:
            sequences.append(cleaned)

    if not sequences:
        raise ValueError(f"No sequences passed max_gap_fraction={max_gap_fraction}.")

    return np.column_stack([seq_to_vec(seq) for seq in sequences]).astype(np.int16, copy=False)


def build_allowed_mutations(
    start_seq: str,
    *,
    mutation_space: str = "unlimited",
    msa_path: str | Path | None = None,
    target_seq: str | None = None,
    custom: dict[int, Sequence[str]] | None = None,
) -> dict[int, list[str]]:
    """Build the position-specific proposal alphabet."""

    start_seq = validate_sequence(start_seq)
    mode = mutation_space.lower()

    if mode == "unlimited":
        return {i: list(AA_ORDER) for i in range(len(start_seq))}

    if mode == "msa":
        if msa_path is None:
            raise ValueError("msa_path is required when mutation_space='msa'.")
        from Bio import AlignIO

        alignment = AlignIO.read(str(msa_path), "fasta")
        if alignment.get_alignment_length() != len(start_seq):
            raise ValueError(
                f"MSA length {alignment.get_alignment_length()} does not match sequence length {len(start_seq)}."
            )
        result: dict[int, list[str]] = {}
        for pos in range(len(start_seq)):
            residues = sorted({str(record.seq[pos]).upper() for record in alignment} & set(AA_ORDER))
            result[pos] = residues
        return result

    if mode == "binary":
        if target_seq is None:
            raise ValueError("target_seq is required when mutation_space='binary'.")
        target_seq = validate_sequence(target_seq)
        if len(target_seq) != len(start_seq):
            raise ValueError("start_seq and target_seq must have equal length for binary mutation space.")
        return {i: sorted({start_seq[i], target_seq[i]} & set(AA_ORDER)) for i in range(len(start_seq))}

    if mode == "custom":
        if custom is None:
            raise ValueError("custom mutation dictionary is required when mutation_space='custom'.")
        result = {}
        for i in range(len(start_seq)):
            residues = [aa.upper() for aa in custom.get(i, ()) if aa.upper() in AA_ORDER]
            result[i] = list(dict.fromkeys(residues))
        return result

    if mode == "lg":
        # LG chooses amino acids from its own matrix; this dictionary is only a
        # compatibility placeholder.
        return {i: list(AA_ORDER) for i in range(len(start_seq))}

    raise ValueError(f"Unsupported mutation_space: {mutation_space!r}")


@contextmanager
def prepend_sys_path(path: str | Path | None) -> Iterator[None]:
    """Temporarily prepend a repository path without changing ``cwd``."""

    if path is None:
        yield
        return
    value = str(Path(path).expanduser().resolve())
    already_present = value in sys.path
    if not already_present:
        sys.path.insert(0, value)
    try:
        yield
    finally:
        if not already_present:
            try:
                sys.path.remove(value)
            except ValueError:
                pass


def torch_device(device: str | None = None):
    """Resolve a PyTorch device, falling back to CPU when CUDA is unavailable."""

    import torch

    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return requested


# ---------------------------------------------------------------------------
# ESM inverse folding helpers
# ---------------------------------------------------------------------------

def run_model(coords, sequence, model, alphabet, cmplx: bool = False, chain_target: str = "A"):
    """Return ESM-IF token probabilities for a sequence on fixed coordinates."""

    del cmplx, chain_target  # retained for backward-compatible call signatures
    import torch
    from esm.inverse_folding.util import CoordBatchConverter

    device = next(model.parameters()).device
    batch_converter = CoordBatchConverter(alphabet)
    coords_t, confidence, _, tokens, padding_mask = batch_converter([(coords, None, sequence)], device=device)
    prev_output_tokens = tokens[:, :-1].to(device)
    logits, _ = model.forward(coords_t, padding_mask, confidence, prev_output_tokens)
    return torch.softmax(torch.swapaxes(logits, 1, 2), dim=-1)


def score_variants(sequence, token_probs, alphabet):
    """Return the model probability assigned to each residue in ``sequence``."""

    aa_labels: list[str] = []
    scores: list[float] = []
    for i, aa in enumerate(sequence):
        aa_labels.append(f"{aa}{i + 1}")
        token_index = alphabet.get_idx(aa)
        scores.append(float(token_probs[0, i, token_index].item()))
    return aa_labels, scores


def masked_absolute(mut, idx, token_probs, alphabet):
    """Backward-compatible single-position probability accessor."""

    return float(token_probs[0, idx, alphabet.get_idx(mut)].item())


# ---------------------------------------------------------------------------
# ProGen helpers
# ---------------------------------------------------------------------------

def create_model(ckpt, fp16: bool = True):
    from progen.modeling_progen import ProGenForCausalLM
    import torch

    kwargs = {"low_cpu_mem_usage": True}
    if fp16:
        kwargs.update({"revision": "float16", "torch_dtype": torch.float16})
    return ProGenForCausalLM.from_pretrained(ckpt, **kwargs)


def create_tokenizer_custom(file):
    from tokenizers import Tokenizer

    with open(file, "r", encoding="utf-8") as handle:
        return Tokenizer.from_str(handle.read())


def cross_entropy(logits, target, reduction: str = "mean"):
    import torch

    return torch.nn.functional.cross_entropy(logits, target, reduction=reduction)


def log_likelihood(logits, target, reduction: str = "mean"):
    return -cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1), reduction=reduction)


def ll(tokens, model, tokenizer, f=log_likelihood, reduction: str = "mean"):
    """Calculate ProGen log-likelihood without hard-coding CUDA."""

    import torch

    device = next(model.parameters()).device
    target = torch.tensor(tokenizer.encode(tokens).ids, device=device)
    autocast_enabled = device.type == "cuda"
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=autocast_enabled):
        logits = model(target, labels=target).logits
        logits = logits[:-1, ...]
        shifted_target = target[1:]

        bos_token, eos_token = 3, 4
        if shifted_target.numel() and int(shifted_target[-1]) in {bos_token, eos_token}:
            logits = logits[:-1, ...]
            shifted_target = shifted_target[:-1]

        if torch.any(shifted_target == bos_token) or torch.any(shifted_target == eos_token):
            raise ValueError("Unexpected terminal token inside ProGen sequence.")

        first_token, last_token = 5, 29
        logits = logits[:, first_token : last_token + 1]
        shifted_target = shifted_target - first_token
        return float(f(logits=logits, target=shifted_target, reduction=reduction).item())


def ce(tokens, model, tokenizer):
    return -ll(tokens, model, tokenizer, f=log_likelihood, reduction="mean")


# ---------------------------------------------------------------------------
# ProteinMPNN / Rosetta helpers
# ---------------------------------------------------------------------------

def one_hot_encode_sequence(seq: str) -> np.ndarray:
    alphabet = AA_ORDER + "X"
    aa_to_index = {aa: idx for idx, aa in enumerate(alphabet)}
    one_hot = np.zeros((len(seq), len(alphabet)), dtype=np.int8)
    for i, aa in enumerate(seq):
        one_hot[i, aa_to_index.get(aa, aa_to_index["X"])] = 1
    return one_hot


def initialize_pyrosetta(options: str = "-out:level 0 -mute core.pack.pack_rotamers core.pack.task"):
    """Initialize PyRosetta once, and only when a Rosetta scorer is requested."""

    import pyrosetta

    if not pyrosetta.rosetta.basic.was_init_called():
        pyrosetta.init(options)
    return pyrosetta


def build_model(path: str | Path, scorefxn: str = "beta_nov16_cart", *, init_options: str | None = None):
    """Clean, relax, and score a PDB structure."""

    pyrosetta = initialize_pyrosetta(init_options or "-out:level 0 -mute core.pack.pack_rotamers core.pack.task")
    from pyrosetta import Pose, pose_from_pdb
    from pyrosetta.toolbox import cleanATOM

    path = str(path)
    cleanATOM(path)
    clean_path = path.replace(".pdb", ".clean.pdb")
    pose = pose_from_pdb(clean_path)
    score_function = pyrosetta.create_score_function(scorefxn)

    relax = pyrosetta.rosetta.protocols.relax.FastRelax()
    relax.constrain_relax_to_start_coords(True)
    relax.coord_constrain_sidechains(True)
    relax.ramp_down_constraints(False)
    relax.cartesian(True)
    relax.min_type("lbfgs_armijo_nonmonotone")
    relax.set_scorefxn(score_function)
    relax.apply(pose)
    energy = float(score_function(pose))
    pose.dump_pdb(path.replace(".pdb", ".relax.pdb"))
    return pose, energy


def clone_rosetta_pose(pose):
    initialize_pyrosetta()
    from pyrosetta import Pose

    clone = Pose()
    clone.assign(pose)
    return clone
