from __future__ import annotations

import math

import numpy as np

from seqsimfit.fitness import proxy2fitness
from seqsimfit.model import (
    accelerated_birthdeath_fixation_prob,
    kimura_fixation_prob,
    mutator,
)
from seqsimfit.simulator import SequenceSimulator
from seqsimfit.utils import build_allowed_mutations


def test_proxy2fitness_is_stable_and_monotonic():
    assert proxy2fitness(0.0, 0.0, 1.0) == 0.5
    assert proxy2fitness(-10000.0, 0.0, 1.0) > 0.999999
    assert proxy2fitness(10000.0, 0.0, 1.0) < 1e-300
    assert proxy2fitness(-1.0, 0.0, 1.0) > proxy2fitness(1.0, 0.0, 1.0)


def test_kimura_neutral_limit():
    assert math.isclose(kimura_fixation_prob(0.5, 0.5, 100), 1 / 200)


def test_accelerated_birthdeath_direction():
    assert accelerated_birthdeath_fixation_prob(0.6, 0.5, 100) == 1.0
    assert 0.0 <= accelerated_birthdeath_fixation_prob(0.4, 0.5, 100) < 1.0


def test_mutator_never_returns_noop():
    allowed = {0: ["A"], 1: ["C", "D"]}
    mutant, pos, old, new = mutator("AC", allowed)
    assert pos == 1
    assert old == "C"
    assert new == "D"
    assert mutant == "AD"


def test_binary_mutation_space():
    allowed = build_allowed_mutations("ACD", mutation_space="binary", target_seq="AED")
    assert allowed == {0: ["A"], 1: ["C", "E"], 2: ["D"]}


def test_callable_simulator_and_repeat_calls_are_additional():
    simulator = SequenceSimulator(
        start_seq="ACDE",
        scorer_configs=[
            {
                "name": "toy",
                "mode": "callable",
                "score_fn": lambda seq: float(seq.count("A")),
                "reference_value": 2.0,
            }
        ],
        Ne=1,  # accelerated model then accepts every valid proposal
        seed=3,
        verbose=False,
    )
    first = simulator.simulate(3)
    second = simulator.simulate(2)
    assert first.accepted_steps == 3
    assert second.accepted_steps == 2
    assert simulator.complete_step == 5
    assert len(simulator.accepted_seq_list) == 6


def test_multi_scorer_weights_are_normalized():
    simulator = SequenceSimulator(
        start_seq="ACDE",
        scorer_configs=[
            {"name": "a", "mode": "callable", "score_fn": lambda seq: 0.0, "weight": 1.0},
            {"name": "b", "mode": "callable", "score_fn": lambda seq: 1.0, "weight": 3.0},
        ],
        verbose=False,
    )
    assert np.isclose(simulator.model_weights["a"], 0.25)
    assert np.isclose(simulator.model_weights["b"], 0.75)


class Node:
    def __init__(self, name, dist=0.0, children=()):
        self.name = name
        self.dist = dist
        self.children = list(children)


def test_tree_siblings_are_independent_and_root_is_restored():
    tree = Node("root", children=[Node("left", 0.25), Node("right", 0.25)])
    simulator = SequenceSimulator(
        start_seq="ACDE",
        scorer_configs=[
            {"name": "toy", "mode": "callable", "score_fn": lambda seq: 0.0}
        ],
        Ne=1,
        seed=11,
        verbose=False,
    )
    sequences = simulator.simulate_on_tree(tree)
    assert set(sequences) == {"root", "left", "right"}
    assert simulator.updated_seq == "ACDE"
    assert simulator.complete_step == 0
