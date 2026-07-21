"""Unified fitness-constrained protein sequence simulator.

This file consolidates the capabilities that were previously split across
``simulator2`` through ``simulator7``:

* single- and multi-model fitness;
* arbitrary per-model weights and reference ratios;
* ProGen + ProteinMPNN combinations;
* multiple ProteinMPNN backbones through named scorer instances;
* tied homo-oligomer ProteinMPNN profiles;
* uniform, MSA-restricted, binary, custom, and LG mutation proposals;
* simulation on phylogenetic trees;
* deterministic random seeds and structured results;
* backward-compatible attribute aliases used by the old notebooks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence
import copy
import random

import numpy as np

try:  # package imports
    from .model import Mutation, fixation_probability, lg_mutator, mutator
    from .scorers import BaseScorer, ScorerConfig, ScoreEvaluation, build_scorer
    from .utils import build_allowed_mutations, validate_sequence
except ImportError:  # flat-file compatibility
    from model import Mutation, fixation_probability, lg_mutator, mutator
    from scorers import BaseScorer, ScorerConfig, ScoreEvaluation, build_scorer
    from utils import build_allowed_mutations, validate_sequence


@dataclass
class EvaluationResult:
    sequence: str
    combined_fitness: float
    component_fitness: dict[str, float]
    proxies: dict[str, float]
    proposed_states: dict[str, Any] = field(repr=False)


@dataclass
class ProposalRecord:
    proposal_index: int
    accepted_index_before: int
    mutation: Mutation
    old_fitness: float
    new_fitness: float
    acceptance_probability: float
    accepted: bool


@dataclass
class SimulationResult:
    start_sequence: str
    final_sequence: str
    requested_accepted_steps: int
    accepted_steps: int
    proposals: int
    terminated_by: str
    final_fitness: float
    final_component_fitness: dict[str, float]
    final_proxies: dict[str, float]

    @property
    def acceptance_rate(self) -> float:
        return self.accepted_steps / self.proposals if self.proposals else 0.0


@dataclass
class _Snapshot:
    sequence: str
    fitness: float
    component_fitness: dict[str, float]
    proxies: dict[str, float]
    scorer_states: dict[str, Any]


class SequenceSimulator:
    """Fitness-constrained origin-fixation sequence simulator.

    New code should prefer ``scorer_configs``. Each configuration is a mapping
    with a unique ``name``, a backend ``mode``, and backend-specific options.
    Example::

        scorer_configs = [
            {
                "name": "mpnn_fold_a",
                "mode": "proteinmpnn",
                "weight": 0.5,
                "ref_ratio": 0.5,
                "model_path": "v_48_020.pt",
                "pdb_path": "fold_a.pdb",
                "repo_path": "/path/to/ProteinMPNN",
            },
            {
                "name": "progen",
                "mode": "progen",
                "weight": 0.5,
                "model_path": "/path/to/progen2-medium",
                "tokenizer_path": "/path/to/tokenizer.json",
                "repo_path": "/path/to/progen2",
            },
        ]

    The historical constructor arguments remain supported for one instance of
    each selected ``evaluation_mode``.
    """

    VALID_LEGACY_MODES = {"rosetta", "dca", "esmif", "esm2", "progen", "proteinmpnn"}

    def __init__(
        self,
        start_seq: str,
        Ne: int = 100,
        evaluation_mode: str | Sequence[str] = "rosetta",
        model_weights: Mapping[str, float] | None = None,
        beta: float | Mapping[str, float] = 1.0,
        ref_ratio: float | Mapping[str, float] = 0.5,
        model: str = "birthdeath",
        pdb_path: str | None = None,
        msa_path: str | None = None,
        J_matrix_path: str | None = None,
        H_matrix_path: str | None = None,
        esmif_model_path: str | None = None,
        progen_model_path: str | None = None,
        progen_tokenizer_path: str | None = None,
        mpnn_model_path: str | None = None,
        chain_id: str | None = None,
        pack_radius: float = 10.0,
        scorefxn_name: str = "beta_nov16_cart",
        stop_mode: str = "step",
        mutation_space: str = "unlimited",
        device: str = "auto",
        step: int = 1000,
        step_limit: int = 100000,
        verbose: bool = True,
        *,
        scorer_configs: Sequence[ScorerConfig | Mapping[str, Any]] | None = None,
        progen_repo_path: str | None = None,
        mpnn_repo_path: str | None = None,
        esmif_score_mode: str = "legacy_probability",
        dca_reference_source: str = "start_seq",
        target_seq: str | None = None,
        custom_allowed_mutations: Mapping[int, Sequence[str]] | None = None,
        lg_matrix: np.ndarray | str | Path | None = None,
        normalize_weights: bool = True,
        seed: int | None = None,
        track_proposals: bool = False,
        mcmc_beta: float = 1.0,
    ):
        self.start_seq = validate_sequence(start_seq)
        self.Ne = int(Ne)
        if self.Ne <= 0:
            raise ValueError("Ne must be a positive integer.")
        self.evaluation_mode = evaluation_mode
        self.model = model
        self.mcmc_beta = float(mcmc_beta)
        self.stop_mode = stop_mode  # retained for compatibility/documentation
        self.mutation_space = mutation_space.lower()
        self.device = device
        self.step = int(step)
        self.step_limit = int(step_limit)
        self.verbose = bool(verbose)
        self.track_proposals = bool(track_proposals)
        self.seed = seed
        self._py_rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)
        self.lg_matrix = self._load_lg_matrix(lg_matrix)

        if self.step < 0 or self.step_limit < 0:
            raise ValueError("step and step_limit must be non-negative.")

        self.target_seq = validate_sequence(target_seq) if target_seq is not None else None
        self.allowed_mutation_dict = build_allowed_mutations(
            self.start_seq,
            mutation_space=self.mutation_space,
            msa_path=msa_path,
            target_seq=self.target_seq,
            custom=dict(custom_allowed_mutations) if custom_allowed_mutations is not None else None,
        )

        if scorer_configs is not None:
            configs = self._explicit_configs(
                scorer_configs,
                model_weights=model_weights,
                beta=beta,
                ref_ratio=ref_ratio,
            )
        else:
            configs = self._legacy_configs(
                evaluation_mode=evaluation_mode,
                model_weights=model_weights,
                beta=beta,
                ref_ratio=ref_ratio,
                pdb_path=pdb_path,
                msa_path=msa_path,
                J_matrix_path=J_matrix_path,
                H_matrix_path=H_matrix_path,
                esmif_model_path=esmif_model_path,
                progen_model_path=progen_model_path,
                progen_tokenizer_path=progen_tokenizer_path,
                mpnn_model_path=mpnn_model_path,
                chain_id=chain_id,
                pack_radius=pack_radius,
                scorefxn_name=scorefxn_name,
                device=device,
                progen_repo_path=progen_repo_path,
                mpnn_repo_path=mpnn_repo_path,
                esmif_score_mode=esmif_score_mode,
                dca_reference_source=dca_reference_source,
            )
        if not configs:
            raise ValueError("At least one scorer must be configured.")

        self.scorers: dict[str, BaseScorer] = {}
        for raw_config in configs:
            scorer = build_scorer(raw_config)
            if scorer.name in self.scorers:
                raise ValueError(f"Duplicate scorer name: {scorer.name!r}")
            self.scorers[scorer.name] = scorer

        self._normalize_scorer_weights(normalize_weights)
        self.evaluation_modes = [scorer.mode for scorer in self.scorers.values()]

        initial_evaluations: dict[str, ScoreEvaluation] = {}
        for name, scorer in self.scorers.items():
            if self.verbose:
                print(f"Preparing scorer {name!r} ({scorer.mode})...")
            initial_evaluations[name] = scorer.prepare(self.start_seq)

        self.energy_reference_dict = {name: scorer.threshold for name, scorer in self.scorers.items()}
        self._set_current_from_evaluations(self.start_seq, initial_evaluations)
        self._seed_snapshot = self._snapshot()
        self._reset_history()

        if self.verbose:
            print("All preparations done!")
            print("Initial combined fitness:", self.updated_fitness)
            print("Initial proxies:", self.updated_energy_dict)
            print("Fitness thresholds:", self.energy_reference_dict)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    @staticmethod
    def _value_for(value: float | Mapping[str, float], name: str, mode: str) -> float:
        if isinstance(value, Mapping):
            if name in value:
                return float(value[name])
            if mode in value:
                return float(value[mode])
            raise ValueError(f"Missing value for scorer {name!r} / mode {mode!r}.")
        return float(value)

    def _explicit_configs(
        self,
        scorer_configs: Sequence[ScorerConfig | Mapping[str, Any]],
        *,
        model_weights: Mapping[str, float] | None,
        beta: float | Mapping[str, float],
        ref_ratio: float | Mapping[str, float],
    ) -> list[ScorerConfig | dict[str, Any]]:
        """Apply global constructor values as fallbacks for explicit configs."""

        prepared: list[ScorerConfig | dict[str, Any]] = []
        for raw in scorer_configs:
            if isinstance(raw, ScorerConfig):
                prepared.append(raw)
                continue
            config = dict(raw)
            if "name" not in config or "mode" not in config:
                raise ValueError("Every scorer config requires unique 'name' and backend 'mode'.")
            name = str(config["name"])
            mode = str(config["mode"])
            if "weight" not in config and model_weights is not None:
                if name in model_weights:
                    config["weight"] = float(model_weights[name])
                elif mode in model_weights:
                    config["weight"] = float(model_weights[mode])
            config.setdefault("weight", 1.0)
            config.setdefault("beta", self._value_for(beta, name, mode))
            config.setdefault("ref_ratio", self._value_for(ref_ratio, name, mode))
            prepared.append(config)
        return prepared

    def _legacy_configs(
        self,
        *,
        evaluation_mode,
        model_weights,
        beta,
        ref_ratio,
        pdb_path,
        msa_path,
        J_matrix_path,
        H_matrix_path,
        esmif_model_path,
        progen_model_path,
        progen_tokenizer_path,
        mpnn_model_path,
        chain_id,
        pack_radius,
        scorefxn_name,
        device,
        progen_repo_path,
        mpnn_repo_path,
        esmif_score_mode,
        dca_reference_source,
    ) -> list[dict[str, Any]]:
        modes = [evaluation_mode] if isinstance(evaluation_mode, str) else list(evaluation_mode)
        if not modes:
            raise ValueError("evaluation_mode cannot be empty.")
        invalid = set(modes) - self.VALID_LEGACY_MODES
        if invalid:
            raise ValueError(f"Unsupported evaluation modes: {sorted(invalid)}")

        weights = model_weights or {mode: 1.0 for mode in modes}
        configs: list[dict[str, Any]] = []
        for mode in modes:
            name = mode
            if name in {config["name"] for config in configs}:
                raise ValueError(
                    "Repeated evaluation modes need scorer_configs with unique names "
                    "(for example two ProteinMPNN backbones)."
                )
            if mode not in weights:
                raise ValueError(f"model_weights is missing {mode!r}.")
            config: dict[str, Any] = {
                "name": name,
                "mode": mode,
                "weight": float(weights[mode]),
                "ref_ratio": self._value_for(ref_ratio, name, mode),
                "beta": self._value_for(beta, name, mode),
            }
            if mode == "rosetta":
                config.update(
                    pdb_path=pdb_path,
                    pack_radius=pack_radius,
                    scorefxn_name=scorefxn_name,
                )
            elif mode == "dca":
                config.update(
                    J_matrix_path=J_matrix_path,
                    H_matrix_path=H_matrix_path,
                    msa_path=msa_path,
                    reference_source=dca_reference_source,
                )
            elif mode == "esmif":
                config.update(
                    model_path=esmif_model_path,
                    pdb_path=pdb_path,
                    chain_id=chain_id or "A",
                    score_mode=esmif_score_mode,
                    device=device,
                )
            elif mode == "esm2":
                config.update(device=device)
            elif mode == "progen":
                config.update(
                    model_path=progen_model_path,
                    tokenizer_path=progen_tokenizer_path,
                    repo_path=progen_repo_path,
                    device=device,
                )
            elif mode == "proteinmpnn":
                config.update(
                    model_path=mpnn_model_path,
                    pdb_path=pdb_path,
                    repo_path=mpnn_repo_path,
                    device=device,
                )
            configs.append(config)
        return configs

    def _normalize_scorer_weights(self, normalize: bool) -> None:
        total = sum(scorer.weight for scorer in self.scorers.values())
        if total <= 0:
            raise ValueError("At least one scorer weight must be positive.")
        if normalize:
            for scorer in self.scorers.values():
                scorer.weight /= total
        self.model_weights = {name: scorer.weight for name, scorer in self.scorers.items()}

    @staticmethod
    def _load_lg_matrix(value):
        if value is None:
            return None
        matrix = np.load(value) if isinstance(value, (str, Path)) else np.asarray(value, dtype=float)
        if matrix.shape != (20, 20):
            raise ValueError(f"LG matrix must have shape (20, 20), got {matrix.shape}.")
        return matrix

    # ------------------------------------------------------------------
    # State and evaluation
    # ------------------------------------------------------------------
    def _combined_fitness(self, component_fitness: Mapping[str, float]) -> float:
        return float(sum(self.model_weights[name] * component_fitness[name] for name in self.scorers))

    def _set_current_from_evaluations(self, sequence: str, evaluations: Mapping[str, ScoreEvaluation]) -> None:
        self.updated_seq = sequence
        self.updated_energy_dict = {name: float(value.proxy) for name, value in evaluations.items()}
        self.updated_component_fitness_dict = {name: float(value.fitness) for name, value in evaluations.items()}
        self._scorer_states = {name: value.state for name, value in evaluations.items()}
        self.updated_fitness = self._combined_fitness(self.updated_component_fitness_dict)
        self.updated_energy = self.updated_energy_dict  # simulator6/7 compatibility
        self._refresh_pose_alias()

    def _refresh_pose_alias(self) -> None:
        rosetta_names = [name for name, scorer in self.scorers.items() if scorer.mode == "rosetta"]
        self.updated_pose = self._scorer_states[rosetta_names[0]] if len(rosetta_names) == 1 else None

    def _snapshot(self) -> _Snapshot:
        states = {
            name: self.scorers[name].clone_state(state)
            for name, state in self._scorer_states.items()
        }
        return _Snapshot(
            sequence=self.updated_seq,
            fitness=self.updated_fitness,
            component_fitness=copy.deepcopy(self.updated_component_fitness_dict),
            proxies=copy.deepcopy(self.updated_energy_dict),
            scorer_states=states,
        )

    def _restore_snapshot(self, snapshot: _Snapshot, *, reset_history: bool) -> None:
        self.updated_seq = snapshot.sequence
        self.updated_fitness = snapshot.fitness
        self.updated_component_fitness_dict = copy.deepcopy(snapshot.component_fitness)
        self.updated_energy_dict = copy.deepcopy(snapshot.proxies)
        self.updated_energy = self.updated_energy_dict
        self._scorer_states = {
            name: self.scorers[name].clone_state(state)
            for name, state in snapshot.scorer_states.items()
        }
        self._refresh_pose_alias()
        if reset_history:
            self._reset_history()

    def estimate(self, sequence: str, mutation: Mutation | None = None) -> EvaluationResult:
        """Evaluate a sequence proposal without committing scorer state."""

        sequence = validate_sequence(sequence)
        if len(sequence) != len(self.start_seq):
            raise ValueError("Proposed sequence length differs from start_seq.")

        evaluations: dict[str, ScoreEvaluation] = {}
        for name, scorer in self.scorers.items():
            evaluations[name] = scorer.evaluate(sequence, mutation, self._scorer_states[name])
        component = {name: result.fitness for name, result in evaluations.items()}
        proxies = {name: result.proxy for name, result in evaluations.items()}
        states = {name: result.state for name, result in evaluations.items()}
        return EvaluationResult(
            sequence=sequence,
            combined_fitness=self._combined_fitness(component),
            component_fitness=component,
            proxies=proxies,
            proposed_states=states,
        )

    def estimate_fitness(
        self,
        mutant_seq: str,
        mutation_pos: int | None = None,
        mutation_aa: str | None = None,
        original_aa: str | None = None,
        initial: bool = False,
    ):
        """Backward-compatible wrapper around :meth:`estimate`.

        ``mutation_pos`` may be one-based (the historical Rosetta convention).
        The return value follows simulator6/7: combined fitness, proxy dictionary,
        proposed state payload, and component-fitness dictionary.
        """

        del initial
        mutation = None
        if mutation_pos is not None and mutation_aa is not None and original_aa is not None:
            zero_based = int(mutation_pos) - 1
            mutation = Mutation(zero_based, original_aa, mutation_aa)
        result = self.estimate(mutant_seq, mutation)
        rosetta_states = {
            name: state
            for name, state in result.proposed_states.items()
            if self.scorers[name].mode == "rosetta"
        }
        state_payload = next(iter(rosetta_states.values())) if len(rosetta_states) == 1 else result.proposed_states
        return result.combined_fitness, result.proxies, state_payload, result.component_fitness

    def fixation(self, old_fitness: float, current_fitness: float) -> float:
        return fixation_probability(
            self.model,
            current_fitness,
            old_fitness,
            population_size=self.Ne,
            mcmc_beta=self.mcmc_beta,
        )

    # ------------------------------------------------------------------
    # Mutation and simulation
    # ------------------------------------------------------------------
    def mutate(self, sequence: str, verbose: bool | None = None):
        verbose = self.verbose if verbose is None else verbose
        if self.mutation_space == "lg":
            if self.lg_matrix is None:
                raise ValueError("mutation_space='lg' requires lg_matrix.")
            mutant_sequence, position, original_aa, new_aa = lg_mutator(sequence, self.lg_matrix, self._np_rng)
        else:
            mutant_sequence, position, original_aa, new_aa = mutator(
                sequence,
                self.allowed_mutation_dict,
                self._py_rng,
            )
        mutation = Mutation(position, original_aa, new_aa)
        if verbose:
            print(f"Mutating {position} from {original_aa} to {new_aa}")
        # Preserve the five-value legacy signature.
        return mutant_sequence, mutation.rosetta_position, position, original_aa, new_aa

    def _reset_history(self) -> None:
        self.all_step = 0
        self.complete_step = 0
        self.accepted_seq_list = [self.updated_seq]
        self.accepted_fitness_list = [self.updated_fitness]
        self.accepted_proxy_list = [copy.deepcopy(self.updated_energy_dict)]
        self.accepted_component_fitness_list = [copy.deepcopy(self.updated_component_fitness_dict)]
        # Accepted mutations are recorded regardless of track_proposals so that
        # site-rate and substitution analyses do not need to infer events from
        # sequence strings.
        self.accepted_mutation_list: list[Mutation] = []
        self.proposal_records: list[ProposalRecord] = []

        # Historical misspellings/attribute names retained as live aliases.
        self.accpeted_seq_fitness_list = self.accepted_fitness_list
        self.accpeted_seq_energy_list = self.accepted_proxy_list

    def _commit(self, result: EvaluationResult, mutation: Mutation | None = None) -> None:
        self.updated_seq = result.sequence
        self.updated_fitness = result.combined_fitness
        self.updated_energy_dict = copy.deepcopy(result.proxies)
        self.updated_energy = self.updated_energy_dict
        self.updated_component_fitness_dict = copy.deepcopy(result.component_fitness)
        self._scorer_states = result.proposed_states
        self._refresh_pose_alias()

        self.accepted_seq_list.append(self.updated_seq)
        self.accepted_fitness_list.append(self.updated_fitness)
        self.accepted_proxy_list.append(copy.deepcopy(self.updated_energy_dict))
        self.accepted_component_fitness_list.append(copy.deepcopy(self.updated_component_fitness_dict))
        if mutation is not None:
            self.accepted_mutation_list.append(mutation)
        self.complete_step += 1

    def simulate(
        self,
        step: int | None = None,
        *,
        max_proposals: int | None = None,
    ) -> SimulationResult:
        """Simulate additional accepted substitutions.

        ``step`` counts accepted substitutions, matching the historical project.
        ``step_limit`` is a hard cumulative proposal limit since the last restart.
        """

        requested = self.step if step is None else int(step)
        if requested < 0:
            raise ValueError("step must be non-negative.")
        start_sequence = self.updated_seq
        start_complete = self.complete_step
        start_all = self.all_step
        target_complete = self.complete_step + requested
        proposal_ceiling = self.step_limit
        if max_proposals is not None:
            if max_proposals < 0:
                raise ValueError("max_proposals must be non-negative.")
            proposal_ceiling = min(proposal_ceiling, self.all_step + int(max_proposals))

        while self.complete_step < target_complete and self.all_step < proposal_ceiling:
            old_fitness = self.updated_fitness
            mutant_sequence, _, position, original_aa, new_aa = self.mutate(self.updated_seq, self.verbose)
            mutation = Mutation(position, original_aa, new_aa)
            proposal = self.estimate(mutant_sequence, mutation)
            p_accept = self.fixation(old_fitness, proposal.combined_fitness)
            accepted = self._py_rng.random() < p_accept

            if self.verbose:
                print(
                    f"proposal={self.all_step + 1} accepted_steps={self.complete_step} "
                    f"old_fitness={old_fitness:.6g} new_fitness={proposal.combined_fitness:.6g} "
                    f"pacc={p_accept:.6g} accepted={accepted}"
                )
                print("old proxies:", self.updated_energy_dict)
                print("new proxies:", proposal.proxies)

            if self.track_proposals:
                self.proposal_records.append(
                    ProposalRecord(
                        proposal_index=self.all_step + 1,
                        accepted_index_before=self.complete_step,
                        mutation=mutation,
                        old_fitness=old_fitness,
                        new_fitness=proposal.combined_fitness,
                        acceptance_probability=p_accept,
                        accepted=accepted,
                    )
                )

            if accepted:
                self._commit(proposal, mutation)
            self.all_step += 1

        accepted_steps = self.complete_step - start_complete
        proposals = self.all_step - start_all
        if accepted_steps >= requested:
            reason = "accepted_steps_reached"
        elif self.all_step >= self.step_limit:
            reason = "step_limit_reached"
        else:
            reason = "max_proposals_reached"

        return SimulationResult(
            start_sequence=start_sequence,
            final_sequence=self.updated_seq,
            requested_accepted_steps=requested,
            accepted_steps=accepted_steps,
            proposals=proposals,
            terminated_by=reason,
            final_fitness=self.updated_fitness,
            final_component_fitness=copy.deepcopy(self.updated_component_fitness_dict),
            final_proxies=copy.deepcopy(self.updated_energy_dict),
        )

    def restart(self, seq: str | None = None) -> None:
        """Reset history without reloading expensive models.

        With ``seq=None`` the simulator returns to the original seed, including a
        cloned Rosetta pose. Arbitrary restarts are supported for stateless scorers;
        a Rosetta scorer requires a matching structural state and therefore only
        supports seed restoration here.
        """

        if seq is None or seq == self.start_seq:
            self._restore_snapshot(self._seed_snapshot, reset_history=True)
            return

        seq = validate_sequence(seq)
        if len(seq) != len(self.start_seq):
            raise ValueError("Restart sequence length differs from start_seq.")
        if any(scorer.mode == "rosetta" for scorer in self.scorers.values()):
            raise ValueError(
                "Arbitrary restart with Rosetta is unsafe because the sequence and pose can diverge. "
                "Use restart() for the original seed or restore a saved simulator snapshot."
            )

        result = self.estimate(seq)
        self._commit_without_history(result)
        self._reset_history()

    def _commit_without_history(self, result: EvaluationResult) -> None:
        self.updated_seq = result.sequence
        self.updated_fitness = result.combined_fitness
        self.updated_energy_dict = copy.deepcopy(result.proxies)
        self.updated_energy = self.updated_energy_dict
        self.updated_component_fitness_dict = copy.deepcopy(result.component_fitness)
        self._scorer_states = result.proposed_states
        self._refresh_pose_alias()

    def analyze(self):
        """Return trajectory-level analyses for the complete accepted history."""

        try:
            from .analysis import analyze_simulator
        except ImportError:
            from analysis import analyze_simulator
        return analyze_simulator(self)

    # ------------------------------------------------------------------
    # Tree simulation
    # ------------------------------------------------------------------
    def simulate_on_tree(
        self,
        node,
        root_sequence: str | None = None,
        *,
        mutation_rate: float = 1.0,
        minimum_branch_steps: int = 0,
    ) -> dict[str, str]:
        """Simulate independent descendant branches on an ``ete3``-like tree.

        Branch length is converted to accepted substitutions as
        ``round(branch_length * sequence_length * mutation_rate)``. Each sibling
        starts from an exact cloned snapshot of its parent, including Rosetta pose
        state when present. The simulator is restored to the root state on return.
        """

        if mutation_rate < 0:
            raise ValueError("mutation_rate must be non-negative.")
        if minimum_branch_steps < 0:
            raise ValueError("minimum_branch_steps must be non-negative.")

        if root_sequence is not None and root_sequence != self.updated_seq:
            self.restart(root_sequence)
        root_snapshot = self._snapshot()
        results: dict[str, str] = {}
        branch_results: dict[str, SimulationResult] = {}
        if getattr(node, "name", None):
            results[node.name] = self.updated_seq

        def walk(parent) -> None:
            parent_snapshot = self._snapshot()
            for child_index, child in enumerate(getattr(parent, "children", ())):
                self._restore_snapshot(parent_snapshot, reset_history=True)
                distance = float(getattr(child, "dist", 0.0) or 0.0)
                branch_steps = int(round(distance * len(self.updated_seq) * mutation_rate))
                branch_steps = max(minimum_branch_steps, branch_steps)
                simulation = self.simulate(branch_steps)
                key = getattr(child, "name", None) or f"unnamed_{id(child)}"
                results[key] = self.updated_seq
                branch_results[key] = simulation
                walk(child)
            self._restore_snapshot(parent_snapshot, reset_history=True)

        try:
            walk(node)
        finally:
            self._restore_snapshot(root_snapshot, reset_history=True)
        self.last_tree_branch_results = branch_results
        return results

    def simulate_on_a_tree(self, node, root_sequence: str, mutation_rate: float = 1.0):
        """Historical method name retained as an alias."""

        return self.simulate_on_tree(node, root_sequence, mutation_rate=mutation_rate)

    def fake_simulate_on_a_tree(self, node, root_sequence: str):
        """Propagate one sequence to every named node without mutation."""

        results = {node.name: root_sequence} if getattr(node, "name", None) else {}
        for child in getattr(node, "children", ()):
            results.update(self.fake_simulate_on_a_tree(child, root_sequence))
        return results


# Historical class name retained so old notebooks can use:
#     from simulator import seq_simulator
seq_simulator = SequenceSimulator
