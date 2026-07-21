"""Model-specific proxy scorers used by :mod:`simulator`.

Each scorer exposes the same contract:

1. prepare itself and score the starting sequence;
2. evaluate a proposed sequence without mutating committed state;
3. clone state when branch simulations need independent descendants.

Raw proxy values always follow the convention ``lower is better``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
import copy
import math

import numpy as np

try:  # package import
    from .fitness import (
        esm2_likelihood_calculator,
        esmif_likelihood_calculator,
        multi_plmdca_energy_calculator,
        plmdca_energy_calculator,
        progen_likelihood_caculator,
        proteinmpnn_cce_calculator,
        proxy2fitness,
        rosetta_dg_calculator,
    )
    from .model import Mutation
    from .utils import (
        build_model,
        clone_rosetta_pose,
        create_model,
        create_tokenizer_custom,
        prepend_sys_path,
        read_fasta_alignment,
        seq_to_vec,
        torch_device,
    )
except ImportError:  # flat-file compatibility
    from fitness import (
        esm2_likelihood_calculator,
        esmif_likelihood_calculator,
        multi_plmdca_energy_calculator,
        plmdca_energy_calculator,
        progen_likelihood_caculator,
        proteinmpnn_cce_calculator,
        proxy2fitness,
        rosetta_dg_calculator,
    )
    from model import Mutation
    from utils import (
        build_model,
        clone_rosetta_pose,
        create_model,
        create_tokenizer_custom,
        prepend_sys_path,
        read_fasta_alignment,
        seq_to_vec,
        torch_device,
    )


@dataclass
class ScoreEvaluation:
    proxy: float
    fitness: float
    state: Any = None


@dataclass
class ScorerConfig:
    """Configuration shared by every scorer backend."""

    name: str
    mode: str
    weight: float = 1.0
    ref_ratio: float = 0.5
    beta: float = 1.0
    reference_value: float | None = None
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "ScorerConfig":
        data = dict(mapping)
        reserved = {"name", "mode", "weight", "ref_ratio", "beta", "reference_value", "options"}
        options = dict(data.pop("options", {}))
        for key in list(data):
            if key not in reserved:
                options[key] = data.pop(key)
        return cls(options=options, **data)


class BaseScorer(ABC):
    """Abstract lower-is-better proxy scorer."""

    def __init__(self, config: ScorerConfig):
        self.config = config
        self.name = config.name
        self.mode = config.mode
        self.weight = float(config.weight)
        self.ref_ratio = float(config.ref_ratio)
        self.beta = float(config.beta)
        self.reference_value = config.reference_value
        self.options = dict(config.options)
        self.reference_proxy: float | None = None
        self.threshold: float | None = None

        if not self.name:
            raise ValueError("Scorer name cannot be empty.")
        if not math.isfinite(self.weight) or self.weight < 0:
            raise ValueError(f"Invalid weight for scorer {self.name!r}: {self.weight}")
        if not math.isfinite(self.ref_ratio):
            raise ValueError(f"Invalid ref_ratio for scorer {self.name!r}: {self.ref_ratio}")
        if not math.isfinite(self.beta) or self.beta <= 0:
            raise ValueError(f"beta must be positive for scorer {self.name!r}.")

    def prepare(self, start_seq: str) -> ScoreEvaluation:
        proxy, state = self._prepare(start_seq)
        proxy = float(proxy)
        if not math.isfinite(proxy):
            raise ValueError(f"Initial proxy from scorer {self.name!r} is not finite: {proxy}")
        self.reference_proxy = proxy
        self.threshold = float(self.reference_value) if self.reference_value is not None else proxy * self.ref_ratio
        return ScoreEvaluation(proxy=proxy, fitness=self.proxy_to_fitness(proxy), state=state)

    def evaluate(self, sequence: str, mutation: Mutation | None, state: Any) -> ScoreEvaluation:
        if self.threshold is None:
            raise RuntimeError(f"Scorer {self.name!r} has not been prepared.")
        proxy, proposed_state = self._evaluate(sequence, mutation, state)
        proxy = float(proxy)
        if not math.isfinite(proxy):
            raise ValueError(f"Proxy from scorer {self.name!r} is not finite: {proxy}")
        return ScoreEvaluation(proxy=proxy, fitness=self.proxy_to_fitness(proxy), state=proposed_state)

    def proxy_to_fitness(self, proxy: float) -> float:
        assert self.threshold is not None
        return proxy2fitness(proxy, self.threshold, self.beta)

    def clone_state(self, state: Any) -> Any:
        return copy.deepcopy(state)

    @abstractmethod
    def _prepare(self, start_seq: str) -> tuple[float, Any]:
        raise NotImplementedError

    @abstractmethod
    def _evaluate(self, sequence: str, mutation: Mutation | None, state: Any) -> tuple[float, Any]:
        raise NotImplementedError


class CallableScorer(BaseScorer):
    """Adapter for a user-supplied ``score_fn(sequence) -> lower-is-better proxy``."""

    def __init__(self, config: ScorerConfig):
        super().__init__(config)
        score_fn = self.options.get("score_fn")
        if not callable(score_fn):
            raise ValueError("Callable scorer requires options['score_fn'].")
        self.score_fn: Callable[[str], float] = score_fn

    def _prepare(self, start_seq: str) -> tuple[float, Any]:
        return float(self.score_fn(start_seq)), None

    def _evaluate(self, sequence: str, mutation: Mutation | None, state: Any) -> tuple[float, Any]:
        del mutation, state
        return float(self.score_fn(sequence)), None


class DcaScorer(BaseScorer):
    def _prepare(self, start_seq: str) -> tuple[float, Any]:
        import h5py

        j_path = self.options.get("J_matrix_path") or self.options.get("j_matrix_path")
        h_path = self.options.get("H_matrix_path") or self.options.get("h_matrix_path")
        if not j_path or not h_path:
            raise ValueError("DCA scorer requires J_matrix_path and H_matrix_path.")
        with h5py.File(j_path, "r") as handle:
            self.J_matrix = handle["J"][:]
        with h5py.File(h_path, "r") as handle:
            self.H_matrix = handle["H"][:]

        source = self.options.get("reference_source", "start_seq")
        if source == "start_seq":
            proxy = plmdca_energy_calculator(seq_to_vec(start_seq), self.J_matrix, self.H_matrix)
        elif source == "msa_first":
            msa_path = self.options.get("msa_path")
            if not msa_path:
                raise ValueError("reference_source='msa_first' requires msa_path.")
            matrix = read_fasta_alignment(msa_path)
            proxy = float(multi_plmdca_energy_calculator(matrix, self.J_matrix, self.H_matrix)[0])
        else:
            raise ValueError("DCA reference_source must be 'start_seq' or 'msa_first'.")
        return proxy, None

    def _evaluate(self, sequence: str, mutation: Mutation | None, state: Any) -> tuple[float, Any]:
        del mutation, state
        return plmdca_energy_calculator(seq_to_vec(sequence), self.J_matrix, self.H_matrix), None


class EsmIfScorer(BaseScorer):
    def _prepare(self, start_seq: str) -> tuple[float, Any]:
        import esm
        from esm.inverse_folding.util import extract_coords_from_structure, load_structure

        model_path = self.options.get("model_path") or self.options.get("esmif_model_path")
        pdb_path = self.options.get("pdb_path")
        self.chain_id = self.options.get("chain_id", "A")
        self.score_mode = self.options.get("score_mode", "legacy_probability")
        self.device = torch_device(self.options.get("device", "auto"))
        if not model_path or not pdb_path:
            raise ValueError("ESM-IF scorer requires model_path and pdb_path.")

        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_path)
        self.model.eval().to(self.device).requires_grad_(False)
        structure = load_structure(pdb_path, self.chain_id)
        self.coords, pdb_sequence = extract_coords_from_structure(structure)
        if len(start_seq) != len(pdb_sequence):
            raise ValueError(
                f"ESM-IF coordinate length {len(pdb_sequence)} does not match start_seq length {len(start_seq)}."
            )
        proxy = esmif_likelihood_calculator(
            self.coords,
            start_seq,
            self.model,
            self.alphabet,
            self.chain_id,
            score_mode=self.score_mode,
        )
        return proxy, None

    def _evaluate(self, sequence: str, mutation: Mutation | None, state: Any) -> tuple[float, Any]:
        del mutation, state
        proxy = esmif_likelihood_calculator(
            self.coords,
            sequence,
            self.model,
            self.alphabet,
            self.chain_id,
            score_mode=self.score_mode,
        )
        return proxy, None


class Esm2Scorer(BaseScorer):
    def _prepare(self, start_seq: str) -> tuple[float, Any]:
        import esm

        self.device = torch_device(self.options.get("device", "auto"))
        model_name = self.options.get("model_name", "esm2_t33_650M_UR50D")
        loader = getattr(esm.pretrained, model_name, None)
        if loader is None:
            raise ValueError(f"Unknown esm.pretrained loader: {model_name!r}")
        self.model, self.alphabet = loader()
        self.model.eval().to(self.device).requires_grad_(False)
        self.batch_converter = self.alphabet.get_batch_converter()
        proxy = esm2_likelihood_calculator(
            start_seq,
            self.model,
            self.alphabet,
            self.batch_converter,
            device=self.device,
        )
        return proxy, None

    def _evaluate(self, sequence: str, mutation: Mutation | None, state: Any) -> tuple[float, Any]:
        del mutation, state
        return (
            esm2_likelihood_calculator(
                sequence,
                self.model,
                self.alphabet,
                self.batch_converter,
                device=self.device,
            ),
            None,
        )


class ProGenScorer(BaseScorer):
    def _prepare(self, start_seq: str) -> tuple[float, Any]:
        self.device = torch_device(self.options.get("device", "auto"))
        model_path = self.options.get("model_path") or self.options.get("progen_model_path")
        tokenizer_path = self.options.get("tokenizer_path") or self.options.get("progen_tokenizer_path")
        repo_path = self.options.get("repo_path") or self.options.get("progen_repo_path")
        if not model_path or not tokenizer_path:
            raise ValueError("ProGen scorer requires model_path and tokenizer_path.")

        fp16 = bool(self.options.get("fp16", self.device.type == "cuda"))
        with prepend_sys_path(repo_path):
            self.model = create_model(model_path, fp16=fp16).to(self.device)
            self.tokenizer = create_tokenizer_custom(tokenizer_path)
        self.model.eval().requires_grad_(False)
        return progen_likelihood_caculator(start_seq, self.model, self.tokenizer), None

    def _evaluate(self, sequence: str, mutation: Mutation | None, state: Any) -> tuple[float, Any]:
        del mutation, state
        return progen_likelihood_caculator(sequence, self.model, self.tokenizer), None


class ProteinMpnnScorer(BaseScorer):
    """Fixed-backbone ProteinMPNN profile scorer.

    Multiple backbones are represented by multiple scorer configurations with
    unique names. This replaces the hard-coded ``pdb_path1/pdb_path2`` branches.

    For a tied homo-oligomer, pass for example::

        {"tied_chains": ["A", "B"], "score_chains": ["A", "B"]}

    The selected chain log-profiles are summed and renormalized, matching the
    behavior introduced in ``simulator5.py``.
    """

    def _prepare(self, start_seq: str) -> tuple[float, Any]:
        import torch
        from scipy.special import logsumexp

        checkpoint_path = self.options.get("model_path") or self.options.get("mpnn_model_path")
        pdb_path = self.options.get("pdb_path")
        repo_path = self.options.get("repo_path") or self.options.get("mpnn_repo_path")
        self.device = torch_device(self.options.get("device", "auto"))
        if not checkpoint_path or not pdb_path:
            raise ValueError("ProteinMPNN scorer requires model_path and pdb_path.")

        with prepend_sys_path(repo_path):
            from protein_mpnn_utils import ProteinMPNN, StructureDatasetPDB, parse_PDB, tied_featurize

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model = ProteinMPNN(
            node_features=128,
            edge_features=128,
            hidden_dim=128,
            num_encoder_layers=3,
            num_decoder_layers=3,
            augment_eps=0.0,
            num_letters=21,
            k_neighbors=checkpoint["num_edges"],
        )
        self.model.to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval().requires_grad_(False)

        pdb_dict_list = parse_PDB(pdb_path)
        if not pdb_dict_list:
            raise ValueError(f"ProteinMPNN parse_PDB returned no structures for {pdb_path}")
        pdb_entry = pdb_dict_list[0]
        dataset = StructureDatasetPDB(pdb_dict_list, truncate=None, max_length=int(self.options.get("max_length", 10000)))

        all_chains = [key[-1:] for key in pdb_entry if key.startswith("seq_chain")]
        designed_chains = list(self.options.get("designed_chains") or all_chains)
        fixed_chains = list(self.options.get("fixed_chains") or [c for c in all_chains if c not in designed_chains])
        chain_id_dict = {pdb_entry["name"]: (designed_chains, fixed_chains)}

        tied_chains = list(self.options.get("tied_chains") or [])
        tied_positions_dict = None
        if tied_chains:
            lengths = [len(pdb_entry[f"seq_chain_{chain}"]) for chain in tied_chains]
            if len(set(lengths)) != 1:
                raise ValueError(f"Tied ProteinMPNN chains have unequal lengths: {dict(zip(tied_chains, lengths))}")
            if lengths[0] != len(start_seq):
                raise ValueError(
                    f"Tied chain length {lengths[0]} does not match start_seq length {len(start_seq)}."
                )
            tied_positions_dict = {
                pdb_entry["name"]: [
                    {chain: [position] for chain in tied_chains}
                    for position in range(1, lengths[0] + 1)
                ]
            }

        seed = int(self.options.get("seed", 42))
        profile = None
        chain_order = None
        chain_lengths = None
        for protein in dataset:
            with torch.no_grad():
                batch_clones = [copy.deepcopy(protein)]
                outputs = tied_featurize(
                    batch_clones,
                    self.device,
                    chain_id_dict,
                    None,
                    None,
                    tied_positions_dict,
                    None,
                    None,
                )
                (
                    X,
                    S,
                    mask,
                    lengths,
                    chain_M,
                    chain_encoding_all,
                    chain_list_list,
                    visible_list_list,
                    masked_list_list,
                    masked_chain_length_list_list,
                    chain_M_pos,
                    omit_AA_mask,
                    residue_idx,
                    dihedral_mask,
                    tied_pos_list_of_lists_list,
                    pssm_coef,
                    pssm_bias,
                    pssm_log_odds_all,
                    bias_by_res_all,
                    tied_beta,
                ) = outputs
                del (
                    lengths,
                    visible_list_list,
                    masked_list_list,
                    masked_chain_length_list_list,
                    omit_AA_mask,
                    dihedral_mask,
                    tied_pos_list_of_lists_list,
                    pssm_coef,
                    pssm_bias,
                    pssm_log_odds_all,
                    bias_by_res_all,
                    tied_beta,
                )
                torch.manual_seed(seed)
                randn = torch.zeros(chain_M.shape, device=X.device)
                log_probs = self.model.conditional_probs(
                    X,
                    S,
                    mask,
                    chain_M * chain_M_pos,
                    residue_idx,
                    chain_encoding_all,
                    randn,
                )
                profile = log_probs.detach().cpu().numpy()[0]
                chain_order = list(chain_list_list[0])
                chain_lengths = [len(pdb_entry[f"seq_chain_{chain}"]) for chain in chain_order]
            break

        if profile is None or chain_order is None or chain_lengths is None:
            raise RuntimeError("ProteinMPNN dataset produced no profile.")
        if sum(chain_lengths) != profile.shape[0]:
            raise ValueError(
                f"ProteinMPNN chain lengths {chain_lengths} do not sum to profile length {profile.shape[0]}."
            )

        slices: dict[str, slice] = {}
        offset = 0
        for chain, length in zip(chain_order, chain_lengths):
            slices[chain] = slice(offset, offset + length)
            offset += length

        score_chains = list(self.options.get("score_chains") or [])
        if not score_chains:
            if profile.shape[0] == len(start_seq):
                selected_profile = profile
            elif tied_chains:
                score_chains = tied_chains
            else:
                matching = [chain for chain in chain_order if chain_lengths[chain_order.index(chain)] == len(start_seq)]
                if len(matching) == 1:
                    score_chains = matching
                else:
                    raise ValueError(
                        "ProteinMPNN structure contains multiple chains. Set score_chains explicitly, "
                        "or configure tied_chains for a tied homo-oligomer."
                    )

        if score_chains:
            missing = [chain for chain in score_chains if chain not in slices]
            if missing:
                raise ValueError(f"Requested ProteinMPNN score_chains not found: {missing}")
            selected = [profile[slices[chain]] for chain in score_chains]
            if any(part.shape[0] != len(start_seq) for part in selected):
                raise ValueError(
                    f"Every selected ProteinMPNN chain must have length {len(start_seq)}; "
                    f"got {[part.shape[0] for part in selected]}."
                )
            if len(selected) == 1:
                selected_profile = selected[0]
            else:
                combine = self.options.get("combine_chains", "sum_log_probs")
                stack = np.stack(selected, axis=0)
                if combine == "sum_log_probs":
                    unnormalized = np.sum(stack, axis=0)
                    selected_profile = unnormalized - logsumexp(unnormalized, axis=-1, keepdims=True)
                elif combine == "mean_log_probs":
                    unnormalized = np.mean(stack, axis=0)
                    selected_profile = unnormalized - logsumexp(unnormalized, axis=-1, keepdims=True)
                else:
                    raise ValueError("combine_chains must be 'sum_log_probs' or 'mean_log_probs'.")

        self.log_profile = np.asarray(selected_profile, dtype=np.float32)
        proxy = proteinmpnn_cce_calculator(start_seq, self.log_profile)
        return proxy, None

    def _evaluate(self, sequence: str, mutation: Mutation | None, state: Any) -> tuple[float, Any]:
        del mutation, state
        return proteinmpnn_cce_calculator(sequence, self.log_profile), None


class RosettaScorer(BaseScorer):
    def _prepare(self, start_seq: str) -> tuple[float, Any]:
        import pyrosetta

        pdb_path = self.options.get("pdb_path")
        if not pdb_path:
            raise ValueError("Rosetta scorer requires pdb_path.")
        self.pack_radius = float(self.options.get("pack_radius", 10.0))
        self.scorefxn_name = self.options.get("scorefxn_name", "beta_nov16_cart")
        init_options = self.options.get("init_options")
        pose, proxy = build_model(pdb_path, self.scorefxn_name, init_options=init_options)
        pose_sequence = pose.sequence()
        if pose_sequence != start_seq:
            raise ValueError(
                "Rosetta relaxed PDB sequence does not match start_seq. "
                "Provide a matching structure or explicitly rebuild it before simulation."
            )
        self.scorefxn = pyrosetta.create_score_function(self.scorefxn_name)
        return proxy, pose

    def _evaluate(self, sequence: str, mutation: Mutation | None, state: Any) -> tuple[float, Any]:
        del sequence
        if mutation is None:
            proxy = float(self.scorefxn(state))
            return proxy, self.clone_state(state)
        _, proxy, proposed_pose = rosetta_dg_calculator(
            state,
            mutation.rosetta_position,
            mutation.original_aa,
            mutation.new_aa,
            self.pack_radius,
            self.scorefxn,
        )
        return proxy, proposed_pose

    def clone_state(self, state: Any) -> Any:
        return clone_rosetta_pose(state)


SCORER_CLASSES = {
    "callable": CallableScorer,
    "custom": CallableScorer,
    "dca": DcaScorer,
    "esmif": EsmIfScorer,
    "esm_if": EsmIfScorer,
    "esm2": Esm2Scorer,
    "progen": ProGenScorer,
    "progen2": ProGenScorer,
    "proteinmpnn": ProteinMpnnScorer,
    "mpnn": ProteinMpnnScorer,
    "rosetta": RosettaScorer,
}


def build_scorer(config: ScorerConfig | Mapping[str, Any]) -> BaseScorer:
    if not isinstance(config, ScorerConfig):
        config = ScorerConfig.from_mapping(config)
    mode = config.mode.lower().replace("-", "_")
    scorer_class = SCORER_CLASSES.get(mode)
    if scorer_class is None:
        raise ValueError(f"Unsupported scorer mode: {config.mode!r}")
    return scorer_class(config)
