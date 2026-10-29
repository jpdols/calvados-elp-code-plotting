# Design Rule 6 — VG/IG/KG architectures

**Analysis:** `plotting_thesis/design_rule_6/analyse_dr6_vgigkg_rg.py` — reads VGIGKG `all_rg_values.csv`; generates per-replicate and across-replicate Rg/kinetic summaries and quality/coverage tables.

**Candidate final thesis plot:** `plot_dr6_architecture_focus_v12_readable_balanced.py` — requires per-replicate summary CSV, outputs main and supplementary figures, figure manifest. The v11 larger-font variant is a close alternative.

**Other useful studies:** `plot_dr6_library_design_overview.py` (methods; no simulation data), `plot_dr6_ratio_split_extended.py`, `plot_dr6_full_library_all_ratios.py`, `plot_dr6_ternary_focused.py`, `plot_dr6_vgigkg_comparisons_updated.py`.

**Review:** scripts contain explicit motif definitions or assumptions; verify against the definitive 84-motif manifests. Archive older style variants only once correct final PDFs are identified.
