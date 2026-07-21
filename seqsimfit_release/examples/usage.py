"""Lightweight example using a custom scorer; no optional ML dependency needed."""

from seqsimfit import SequenceSimulator


def alanine_penalty(sequence: str) -> float:
    """Toy lower-is-better proxy."""
    return float(sequence.count("A"))


simulator = SequenceSimulator(
    start_seq="ACDEFGHIK",
    scorer_configs=[
        {
            "name": "toy",
            "mode": "callable",
            "score_fn": alanine_penalty,
            "weight": 1.0,
            "reference_value": 2.0,
        }
    ],
    Ne=10,
    seed=7,
    verbose=False,
)

result = simulator.simulate(step=10)
print(result)
