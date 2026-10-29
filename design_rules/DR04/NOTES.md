# Design Rule 4 — distributing charge

**Primary upstream analysis:** `plotting_thesis/design_rule_4/analyse_charge_distribution_rg_v3.py` — reads VGEG, VGKG and IGKG `all_rg_values.csv`; outputs `rg_summary_VGEG.csv`, `rg_summary_VGKG.csv`, `rg_summary_IGKG.csv` and `rg_summary_all_libraries.csv` plus diagnostics.

**Candidate final thesis plot:** `plotting_thesis/design_rule_4/plot_design_rule4_thesis_v20_final.py` — reads combined `rg_summary_all_libraries.csv`, writes print figures and normalized CSV tables.

**Additional candidates:** `plot_design_rule4_regime_overview_v2.py`, `plot_design_rule4_architecture_average_v6.py`. Versions v5–v19 of the thesis figure should be reviewed/archived, particularly v11 (different figure design).

**Important:** upstream analysis default threshold is 4.61 nm; preserve actual processing settings and timing. Do not interchange raw rate, log-ratio and percentage-point normalization.
