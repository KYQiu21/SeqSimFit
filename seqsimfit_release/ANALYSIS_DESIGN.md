# SeqSimFit trajectory-analysis design

## Source material reviewed

The analysis layer was distilled from the manuscript and the exploratory
notebooks `examine_aa_comp.ipynb`, `examine_evol_rate.ipynb`, and
`examine_sub.ipynb`.

## Analyses included in the package

1. **Trajectory divergence**
   - Hamming distance and sequence identity to the starting sequence.
   - Selection of the first/closest sequence at a target identity.

2. **Site-specific evolutionary rate**
   - Raw accepted substitution counts at each position.
   - Aggregate mean, sum, or median profiles over replicate trajectories.
   - Pearson or Spearman comparison with external site-rate profiles.

3. **Amino-acid distributions**
   - Position-specific counts, frequencies, and alignment occupancy.
   - Global amino-acid composition.
   - Site-wise and occupancy-weighted KL divergence from a reference alignment.

4. **Substitution patterns**
   - Directed substitution counts.
   - State exposure/sojourn counts.
   - Unit-rate-normalized continuous-time rate matrix.
   - Symmetric exchangeability matrix and comparison with LG/DCA references.

5. **I/O and alignment helpers**
   - Lightweight FASTA reading/writing without a required Biopython dependency.
   - Query-gap removal and optional target-region trimming.

## Exploratory notebook behavior intentionally corrected

### KL direction and normalization

The notebook helper named its first input `obs` but called it with simulated
trajectories, making the effective direction opposite to the manuscript formula.
It also divided the weighted site sum by the number of simulated sequence states,
which is not a standard KL normalization. The formal function therefore makes the
direction explicit as `D_KL(reference || simulated)` and exposes three named
normalizations: `sum`, `mean`, and `length`.

### Distinct residues versus substitution rate

The number of distinct residue types visited at a site ignores reversions and
repeat substitutions. It remains available as `site_residue_diversity`, while
`site_substitution_counts` implements the manuscript rate proxy by counting every
accepted event.

### Soft-profile EMD

The exploratory Wasserstein calculation treated the 20 probability values as
ordered samples. Amino-acid categories do not possess the required one-dimensional
metric ordering, so this quantity was not promoted into the formal package.
Jensen-Shannon analysis for future AF-back profile trajectories can be added
separately with an explicit alphabet and probability interpretation.

### Substitution model estimation

The package uses the later, more principled notebook estimator:
`q_ij = N_ij / S_i`, followed by unit expected-rate normalization and symmetric
exchangeability calculation. The earlier log-odds matrix helper is not used as
the default evolutionary rate-matrix estimator.

### Alignment trimming

The notebook helper depended on a global `consurf_seq` variable despite accepting
a sequence argument. `trim_alignment_to_query` is pure and uses only its explicit
arguments.

## Deferred analysis

Residue-residue covariation is not included in the first Colab-facing analysis
layer because the manuscript workflow requires phylogenetically structured
simulated alignments and external plmDCA inference. It should be presented as an
advanced workflow rather than a one-click analysis of a single forward
trajectory.
