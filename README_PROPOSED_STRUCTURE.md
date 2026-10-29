# Proposed analysis and plotting repository structure

**Status: proposal for manual review, not a validated workflow.** No original scripts have been moved or deleted. The current `main` branch remains an untouched audit snapshot.

## Layout

- `model_validation/`: temperature response, reference ELPs, LCST, gamma and lambda scaling.
- `design_rules/DR01` through `DR08`: per-rule analysis and figure scripts, with supplementary figures kept near their parent rule.
- `supplementary/`: methods graphics and cross-rule analyses.
- `utilities/`: general Rg and kinetic calculations, trajectory processing and diagnostics.
- `archive/`: prior iterations and experimental plotting code, retained in original Git history.
- `data/`: lightweight summaries, manifests and links to external frame-resolved/trajectory datasets.

Each folder has a NOTES.md containing **candidate source paths in the original repository**, their expected input data, and review points. Paths are not moved yet; user approval and run verification are needed before any filename is treated as a definitive thesis figure.

## Key upstream gap

The DelftBlue `analyse_library_overview.py` script is not included in this snapshot. It is important for producing `all_rg_values.csv`, `per_run_summary.csv`, `dwell_events.csv`, and `per_architecture_300_vs_370.csv`. Retrieve and preserve its exact version and settings.

## Reproducibility approach

`.dcd` and `.pdb` belong in a separate research data archive; retain a manifest with locations, checksums, parameters, frame time, stride and CALVADOS version. For analyses requiring full frame histories, provide `all_rg_values.csv` externally if large. For plot-only reproduction, include the exact final input summaries where small enough.

## Review priorities

1. Confirm which script generated each figure actually present in the final thesis.
2. Recover missing DelftBlue analysis and contact-processing scripts.
3. Preserve per-analysis threshold, state definition, time step and treatment of censored dwell events.
4. Resolve FASTA/sequence provenance, including any explicitly assumed control sequence positions.
5. Add commands and test each script before classifying as reproducible.
