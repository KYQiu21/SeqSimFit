# SeqSimFit — sequence simulation under fitness constraints

## Installation / use

The folder itself is importable as a package when its parent directory is on `PYTHONPATH`:

```python
from seqsimfit import SequenceSimulator, seq_simulator
```

The individual files also support the old flat layout:

```python
from simulator import seq_simulator
```

when `simulator.py`, `scorers.py`, `fitness.py`, `utils.py`, and `model.py` are in the same directory.

## 1. Single model using the legacy constructor

```python
from seqsimfit import SequenceSimulator

sim = SequenceSimulator(
    start_seq=sequence,
    evaluation_mode="dca",
    J_matrix_path="J.h5",
    H_matrix_path="H.h5",
    ref_ratio=0.5,
    beta=1.0,
    Ne=100,
    seed=1,
    verbose=False,
)

result = sim.simulate(step=500)  # 500 accepted substitutions
print(result.final_sequence, result.acceptance_rate)
```

## 2. ProGen2 + ProteinMPNN

```python
sim = SequenceSimulator(
    start_seq=sequence,
    scorer_configs=[
        {
            "name": "progen",
            "mode": "progen",
            "weight": 0.5,
            "ref_ratio": 0.5,
            "model_path": "/models/progen2-medium",
            "tokenizer_path": "/repos/progen2/tokenizer.json",
            "repo_path": "/repos/progen2",
            "device": "cuda",
        },
        {
            "name": "mpnn",
            "mode": "proteinmpnn",
            "weight": 0.5,
            "ref_ratio": 0.5,
            "model_path": "/repos/ProteinMPNN/vanilla_model_weights/v_48_020.pt",
            "pdb_path": "structure.pdb",
            "repo_path": "/repos/ProteinMPNN",
            "score_chains": ["A"],
            "device": "cuda",
        },
    ],
    mutation_space="unlimited",
    seed=1,
)
```

## 3. Two ProteinMPNN backbones

```python
sim = SequenceSimulator(
    start_seq=sequence,
    scorer_configs=[
        {
            "name": "mpnn_fold1",
            "mode": "proteinmpnn",
            "weight": 0.7,
            "ref_ratio": 0.5,
            "model_path": mpnn_checkpoint,
            "pdb_path": "fold1.pdb",
            "repo_path": mpnn_repo,
            "score_chains": ["A"],
        },
        {
            "name": "mpnn_fold2",
            "mode": "proteinmpnn",
            "weight": 0.3,
            "ref_ratio": 0.5,
            "model_path": mpnn_checkpoint,
            "pdb_path": "fold2.pdb",
            "repo_path": mpnn_repo,
            "score_chains": ["A"],
        },
    ],
)
```

## 4. Tied dimer + another backbone

```python
sim = SequenceSimulator(
    start_seq=monomer_sequence,
    scorer_configs=[
        {
            "name": "tied_dimer",
            "mode": "proteinmpnn",
            "weight": 0.5,
            "model_path": mpnn_checkpoint,
            "pdb_path": "homodimer.pdb",
            "repo_path": mpnn_repo,
            "designed_chains": ["A", "B"],
            "tied_chains": ["A", "B"],
            "score_chains": ["A", "B"],
            "combine_chains": "sum_log_probs",
        },
        {
            "name": "alternative_fold",
            "mode": "proteinmpnn",
            "weight": 0.5,
            "model_path": mpnn_checkpoint,
            "pdb_path": "alternative.pdb",
            "repo_path": mpnn_repo,
            "score_chains": ["A"],
        },
    ],
)
```

## 5. ESM-IF reproducibility versus new NLL scoring

```python
# Reproduce old behavior
{"name": "esmif", "mode": "esmif", "score_mode": "legacy_probability", ...}

# Recommended for a new experiment
{"name": "esmif", "mode": "esmif", "score_mode": "nll", ...}
```

## 6. Tree simulation

```python
# `tree` can be an ete3 Tree/TreeNode or any object exposing `.name`, `.dist`, `.children`.
sequences = sim.simulate_on_tree(
    tree,
    mutation_rate=1.0,
    minimum_branch_steps=0,
)
branch_diagnostics = sim.last_tree_branch_results
```

Sibling branches are generated from cloned parent states, including cloned Rosetta poses. The simulator returns to the root state after the tree run.

## Main result attributes

The new structured result is returned by `simulate()`. These historical attributes are also updated:

- `updated_seq`
- `updated_fitness`
- `updated_energy_dict`
- `updated_component_fitness_dict`
- `accepted_seq_list`
- `accpeted_seq_fitness_list` (legacy misspelling retained)
- `accpeted_seq_energy_list` (legacy name retained)
- `all_step`: proposed mutations since restart
- `complete_step`: accepted substitutions since restart

## Dependencies

Core tests require only NumPy. Backends load their own optional dependencies lazily:

- BioPython and h5py for MSA/DCA
- PyTorch and fair-esm for ESM models
- the local ProGen2 repository
- the local ProteinMPNN repository
- PyRosetta for Rosetta scoring
- SciPy for tied-chain ProteinMPNN profile normalization

No optional backend is imported merely by importing the simulator.

## Trajectory analysis (v1.1)

SeqSimFit now includes backend-independent trajectory analyses. Every accepted
mutation is recorded in `simulator.accepted_mutation_list`, and the complete
accepted history can be summarized directly:

```python
analysis = simulator.analyze()

analysis.identity_to_start          # one value per accepted sequence state
analysis.hamming_to_start
analysis.site_substitution_counts   # manuscript site-rate proxy
analysis.fitness
analysis.proxies                     # one trajectory per named scorer
analysis.component_fitness
```

The same analysis can be applied to a plain list of sequences:

```python
from seqsimfit import analyze_trajectory
analysis = analyze_trajectory(sequence_trajectory)
```

### Amino-acid distributions and alignment divergence

```python
from seqsimfit import alignment_kl_divergence, read_fasta

natural_msa = read_fasta("natural_alignment.fasta")
kl = alignment_kl_divergence(
    reference_sequences=natural_msa,
    simulated_sequences=simulator.accepted_seq_list,
    normalization="mean",
)

print(kl.aggregate)
print(kl.sitewise)
```

The direction is explicitly `D_KL(natural || simulated)`. Reference-column
non-gap occupancy is used as the site weight. `normalization="sum"` reports the
weighted sum written in the manuscript; `"mean"` reports the weighted mean
across occupied sites and is usually easier to compare across proteins.

### Site-rate recovery

```python
from seqsimfit import correlate_site_rates

simulated_rates = analysis.site_substitution_counts
comparison = correlate_site_rates(simulated_rates, reference_rates)
print(comparison.coefficient)
```

`site_substitution_counts` counts every accepted event, including reversions.
The exploratory distinct-residue statistic remains available under the more
accurate name `site_residue_diversity`.

### Empirical substitution model

```python
from seqsimfit import estimate_substitution_model, compare_exchangeability

model = estimate_substitution_model(replicate_trajectories)
print(model.counts)
print(model.rate_matrix)
print(model.exchangeability)

comparison = compare_exchangeability(model, lg_exchangeability)
print(comparison.coefficient)
```

The estimated off-diagonal rates use `q_ij = N_ij / S_i` and are normalized to
unit expected substitution rate. Symmetric exchangeabilities can be compared
with LG- or DCA-derived reference matrices over amino-acid pairs.

### Select a sequence at a target identity

```python
from seqsimfit import first_state_at_identity

step, sequence, observed_identity = first_state_at_identity(
    simulator.accepted_seq_list,
    target_identity=0.5,
)
```

This is useful for endpoint molecular evaluation at a matched evolutionary
distance.
