# Consolidation review of the historical simulator versions

## Version lineage

| File | Distinct purpose | Kept in formal version |
|---|---|---|
| `simulator2.py` | Stable single-scorer implementation; tree simulation; optional mutation-rate multiplier | Yes |
| `simulator3_mpnnprogen.py` | Hard-coded two-model ProGen + ProteinMPNN combination with two thresholds | Generalized to arbitrary named scorers |
| `simulator4.py` | Two ProteinMPNN backbones (`pdb_path1`, `pdb_path2`) and weighted fitness | Generalized as two ProteinMPNN scorer configs |
| `simulator5.py` | Adds tied A/B homodimer positions and combines chain log-profiles | Preserved through `tied_chains`, `score_chains`, and `combine_chains` |
| `simulator6.py` | First general list-of-models implementation; contains duplicated imports and a complete obsolete class copy in comments | Used as architectural base; dead code removed |
| `simulator7.py` | Same active implementation as simulator6, but imports `fitness7` / `utils7` | Used as latest general baseline |
| `utils (2).py` | Utility functions with PyRosetta imports disabled | Functionality retained, cleaned, and made lazy |
| `utils7.py` | Same utilities, but initializes PyRosetta during import | Initialization moved inside Rosetta-only functions |
| `fitness7.py` | Proxy-to-fitness and six model proxy calculators | Retained with device handling and numerical validation |
| `model (1).py` | Fixation models, uniform mutator, and LG mutator | Retained with stable formulas and no hard-coded LG path |

## Problems fixed

### Architecture and maintenance

1. Wildcard imports made it unclear which file owned each function and hid missing imports.
2. `simulator6.py` and `simulator7.py` contained roughly 600 lines of obsolete commented class code.
3. The same feature was repeatedly implemented by adding numbered files rather than model instances.
4. Dual-backbone and tied-dimer logic were hard-coded into the simulator instead of represented as scorer configuration.
5. Tree simulation disappeared from the latest general versions.

### Runtime side effects and portability

1. `utils7.py` initialized PyRosetta immediately on import.
2. ProGen preparation changed the global working directory with `os.chdir()`.
3. ESM-2, ESM-IF, and ProGen helper code repeatedly forced CUDA instead of respecting the configured device.
4. The LG matrix path was hard-coded and the matrix was loaded on every mutation proposal.
5. ProteinMPNN repository paths were hard-coded in the implementation.

### Correctness and numerical stability

1. Kimura fixation produced `0/0` at exact neutrality; the analytical neutral limit is now used.
2. Fixation probabilities are clamped and protected against zero fitness, overflow, and underflow.
3. `proxy2fitness` could overflow for strong selection; it now uses a stable sigmoid calculation.
4. The old mutator could return a no-op mutation when a site had no alternative; only genuinely mutable positions are now sampled.
5. `simulate_on_a_tree()` multiplied an integer by `mutation_rate` after casting, potentially sending a float to `simulate()`.
6. Zero-length branches were forced to undergo one accepted substitution. The new default permits zero steps; `minimum_branch_steps=1` recreates the old behavior.
7. The MSA parser's dot-handling assertion compared raw length with dot-filtered length and was internally inconsistent.
8. `simulate(step)` formerly treated `step` as an absolute accepted-step target. It now means additional accepted substitutions, which is less error-prone across repeated calls.
9. Arbitrary `restart(seq)` with Rosetta could leave the sequence and pose inconsistent. The formal version refuses that unsafe operation.
10. Sibling tree branches now start from cloned parent scorer state rather than relying on a partially reset global object.

### Scientific-score semantics

1. All proxies are explicitly documented as lower-is-better.
2. ESM-IF historical scoring used `-sum(probability)` rather than negative log-likelihood. It is preserved as `legacy_probability`; conventional `nll` is opt-in.
3. DCA can use either the starting sequence (new default) or the first MSA sequence (`reference_source='msa_first'`) as its reference.
4. ProteinMPNN continues to use a fixed backbone-derived profile, matching the historical simulations. Multiple profiles are independent scorer instances.

## Intentional compatibility decisions

- The class alias `seq_simulator` remains available.
- `accpeted_seq_fitness_list` and `accpeted_seq_energy_list` retain their historical misspellings as aliases.
- `progen_likelihood_caculator` retains its historical spelling, with a correctly spelled alias added.
- `estimate_fitness()` retains the simulator6/7 four-item return structure.
- The default accelerated birth-death model still accepts beneficial and neutral proposals with probability one.
- Default ESM-IF behavior remains the legacy probability score to avoid silently changing existing results.

## Recommended migration

Use `scorer_configs` for all new experiments. Treat each model/backbone as a uniquely named scorer. This removes the need for future `simulator8.py`, `simulator9.py`, and so on: a new combination should be configuration, while a genuinely new scoring model should be a new scorer class.
