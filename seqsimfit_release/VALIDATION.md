# Validation status

## Completed in this environment

- All Python files parsed and passed `python -m compileall`.
- The source distribution built successfully as `seqsimfit-1.0.0-py3-none-any.whl`.
- Eight lightweight unit tests passed:
  1. stable and monotonic proxy-to-fitness transformation;
  2. Kimura neutral limit;
  3. accelerated birth-death direction;
  4. mutation proposals cannot silently return a no-op;
  5. binary mutation-space construction;
  6. callable-scorer simulation and repeated-call step semantics;
  7. multi-scorer weight normalization;
  8. independent sibling branches and root-state restoration on a tree.
- Package import and the historical `from simulator import seq_simulator` wrapper were both verified.

## Not executable here

The following backends require local repositories, checkpoints, structures, licensed software, and/or GPUs that were not included with the uploaded files:

- PyRosetta
- ESM-IF
- ESM-2 model weights
- ProGen2
- ProteinMPNN
- project-specific DCA matrices and MSA files

Their code paths were consolidated and statically checked against the uploaded implementations, but they still need one small end-to-end smoke test in the user's Raven environment before the old numbered versions are archived.

## Version 1.1 trajectory-analysis validation

The analysis extension adds tests for accepted mutation recording, site-rate
counts including reversions, replicate aggregation, position-specific amino-acid
frequencies, gap-weighted reference-to-simulation KL divergence, empirical rate
matrix normalization, exchangeability symmetry, rate correlations, target-
identity sequence selection, FASTA-independent alignment trimming, and direct
`SequenceSimulator.analyze()` integration.

Current result: **17/17 tests passed**.
