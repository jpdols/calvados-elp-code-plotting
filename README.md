# CALVADOS ELP — analysis and plotting (review branch)

This branch reorganizes an MSc research code audit **without modifying the scripts**. Select scripts are copied to thesis-oriented folders; *all original source files* are preserved under `archive/original_snapshot/` with their original relative paths. Older versions and non-analysis simulation source remain in the archive for provenance.

## Navigate

- [Model validation](model_validation/NOTES.md) — reference sequences, temperatures and model scaling.
- [Design Rules 01–08](design_rules/DR01/NOTES.md) — each rule has a `NOTES.md` plus selected `analysis/`, `plotting/` and `supplementary/` files.
- [Supplementary](supplementary/NOTES.md) — methods plots, shared statistics, exploratory library plots.
- [Utilities](utilities/NOTES.md) — reusable Rg, kinetics, property comparisons and trajectory-processing tools.
- [Archive](archive/NOTES.md) — original source snapshot and older versions.
- [Data](data/NOTES.md) — input requirements, missing datasets and provenance needs.
- [Complete migration manifest](MIGRATION_MANIFEST.md) — mapping for every source file.

## Reproducing analyses

1. Locate the relevant script in a design-rule folder or model validation.
2. Consult its `--help` (where available) and the folder's `NOTES.md` for source datasets.
3. For Rg/kinetics, obtain the exact `all_rg_values.csv` and/or derived summary CSV used in the study; for contact/shape work, raw `.dcd` + `.pdb` may be necessary.
4. Run the script and validate its numerical and graphical results against the thesis.

**Status: organized but not yet run-tested.** No complete data or environment distribution is included. Some scripts rely on hardcoded Windows paths and different collapse-state definitions. Do not assume that a large frame-resolved Rg file can be replaced by a compact summary without changing analyses.

The original audit snapshot is on the [main branch](https://github.com/jpdols/calvados-elp-code-audit/tree/main). This branch is available for review via the draft pull request.
