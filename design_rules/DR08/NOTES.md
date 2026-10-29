# Design Rule 8 — VG/IP/KG sequence architectures

**Upstream analysis:** `plotting_thesis/design_rule_8/create_dr8_complete_summary.py` and/or `create_vgipkg_summary_kinetics.py` — reads three replicate `all_rg_values.csv` files, creates `combined_summary_mean_std.csv`, `replicate_summary.csv`, `dwell_events.csv` and other outputs; needs `design_metadata.csv` for plotting.

**Candidate thesis figures:** `plot_dr8_thesis_updated.py`, `plot_dr8_three_subrules.py`, `plot_dr8_complete_story.py`, `plot_dr8_vgipkg_kinetics.py`; older `plot_dr8_thesis.py` for comparison.

**Inputs:** `combined_summary_mean_std.csv`, `design_metadata.csv`, sometimes `replicate_summary.csv`.

**Scientific caution:** `plot_dr8_thesis_updated.py` documents an assumed completion of control 834's FASTA by appending two motifs; verify against source sequence before publishing as independently confirmed.
