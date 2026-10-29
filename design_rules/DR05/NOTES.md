# Design Rule 5 — charged/ternary architectures and SCD

**Analyses:** `plotting_thesis/design_rule_5/analyse_dr5_rg.py` and `analyse_dr5_ternary_rg.py` — use `all_rg_values.csv`, frame timing, threshold; produce `summary_per_replicate.csv`, `summary_across_replicates.csv`, dwell/data-quality files.

**Candidate thesis plot:** `plot_dr5_scd_370K_PRINT_REDESIGNED.py` — four 370 K observables against full-residue sequence charge decoration (SCD). Input: summary CSV plus **actual FASTA recommended**; produces regression/statistics outputs.

**Other plot candidates:** `plot_dr5_ternary_focused_updated.py`, `plot_dr5_ternary_deviations.py`, `plot_dr5_ternary_comparisons.py`, `plot_dr5_scd_labels.py`.

**Review:** SCD script can reconstruct ideal regular sequences if actual FASTA absent; for scientific provenance supply verified sequences. Compare formatting variants `FIXED`, `LABELS_FIXED`, `PRINT_LARGE` before archiving.
