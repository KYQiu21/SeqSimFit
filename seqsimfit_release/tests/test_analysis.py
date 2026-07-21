from __future__ import annotations

import numpy as np

from seqsimfit import SequenceSimulator
from seqsimfit.analysis import (
    aggregate_site_substitution_counts,
    alignment_kl_divergence,
    analyze_trajectory,
    compare_exchangeability,
    correlate_site_rates,
    estimate_substitution_model,
    first_state_at_identity,
    mutation_events,
    position_frequencies,
    site_residue_diversity,
    site_substitution_counts,
    trim_alignment_to_query,
)


def test_mutation_events_and_site_counts_include_reversions():
    trajectory = ["AAAA", "ACAA", "AAAA", "AAAD"]
    events = mutation_events(trajectory)
    assert [event.label for event in events] == ["A2C", "C2A", "A4D"]
    np.testing.assert_array_equal(site_substitution_counts(trajectory), [0, 2, 0, 1])
    np.testing.assert_array_equal(site_residue_diversity(trajectory), [0, 1, 0, 1])


def test_aggregate_site_counts():
    trajectories = [
        ["AAAA", "ACAA", "AAAA"],
        ["AAAA", "AAAD", "AADD"],
    ]
    np.testing.assert_array_equal(
        aggregate_site_substitution_counts(trajectories, reduction="sum"),
        [0, 2, 1, 1],
    )


def test_position_frequencies_and_reference_to_simulated_kl():
    reference = ["AA", "AC"]
    simulated_same = ["AA", "AC"]
    simulated_different = ["CC", "CC"]

    profile = position_frequencies(reference)
    np.testing.assert_allclose(profile.frequencies[0], [1.0] + [0.0] * 19)
    assert profile.occupancy.tolist() == [1.0, 1.0]

    same = alignment_kl_divergence(reference, simulated_same)
    different = alignment_kl_divergence(reference, simulated_different)
    assert same.aggregate == 0.0
    assert different.aggregate > same.aggregate


def test_alignment_kl_gap_weights():
    reference = ["A-", "AC"]
    simulated = ["CC", "CC"]
    result = alignment_kl_divergence(reference, simulated, normalization="sum")
    np.testing.assert_allclose(result.weights, [1.0, 0.5])
    assert np.isfinite(result.aggregate)


def test_substitution_model_is_normalized():
    trajectories = [
        ["AAAA", "CAAA", "CCAA", "ACAA"],
        ["AAAA", "AAAD", "AADD", "ADDD"],
    ]
    model = estimate_substitution_model(trajectories)
    assert model.total_substitutions == 6
    assert model.transitions_checked == 6
    assert model.single_substitution_fraction == 1.0
    assert np.allclose(model.rate_matrix.sum(axis=1), 0.0)
    expected_rate = -np.sum(model.equilibrium_frequencies * np.diag(model.rate_matrix))
    assert np.isclose(expected_rate, 1.0)
    assert np.allclose(model.exchangeability, model.exchangeability.T)


def test_rate_and_exchangeability_correlations():
    rate = correlate_site_rates([0, 1, 2, 3], [0, 2, 4, 6], method="spearman")
    assert np.isclose(rate.coefficient, 1.0)

    trajectory = ["AAAA", "CAAA", "DAAA", "AAAA"]
    model = estimate_substitution_model(trajectory)
    comparison = compare_exchangeability(model, model)
    assert np.isclose(comparison.coefficient, 1.0)


def test_analyze_trajectory_and_identity_selection():
    trajectory = ["AAAA", "CAAA", "CCAA"]
    analysis = analyze_trajectory(
        trajectory,
        fitness=[1.0, 0.9, 0.8],
        proxies={"toy": [0.0, 1.0, 2.0]},
    )
    assert analysis.accepted_steps == 2
    np.testing.assert_allclose(analysis.identity_to_start, [1.0, 0.75, 0.5])
    index, sequence, identity = first_state_at_identity(trajectory, 0.6)
    assert (index, sequence, identity) == (2, "CCAA", 0.5)


def test_trim_alignment_to_query_is_pure():
    alignment = ["A-CDE", "ATC-E", "A-C-E"]
    assert trim_alignment_to_query(alignment) == ("ACDE", "AC-E", "AC-E")
    assert trim_alignment_to_query(alignment, target_sequence="CD") == ("CD", "C-", "C-")


def test_simulator_records_accepted_mutations_and_analyzes():
    simulator = SequenceSimulator(
        start_seq="ACDE",
        scorer_configs=[
            {"name": "toy", "mode": "callable", "score_fn": lambda seq: 0.0}
        ],
        Ne=1,
        seed=5,
        verbose=False,
    )
    simulator.simulate(4)
    assert len(simulator.accepted_mutation_list) == 4
    analysis = simulator.analyze()
    assert analysis.accepted_steps == 4
    assert int(analysis.site_substitution_counts.sum()) == 4
