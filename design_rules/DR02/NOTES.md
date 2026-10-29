# Design Rule 2 — motif-pair architecture comparison

**Candidate thesis plot:** `plotting_thesis/design_rule_2/plot_DR2_thesis_final_v14_MUTED_GREEN.py` (not independently matched to submitted PDF).

**Analyses:** `analyse_triplicate_statistics_12pairs.py` (three `per_run_summary.csv` replicates; statistical outputs); `analyse_rg_distribution_overlap.py` (frame-resolved `all_rg_values.csv`, pairwise distribution overlap).

**Other/secondary figures:** `plot_DR2_conformational_regime.py`, `plot_design_rule_2_architecture_contrast.py`, `supplementary.py`. Versions v10–v13 of the final figure are review/archive candidates.

**Key inputs:** `per_run_summary.csv`, `all_rg_values.csv`; some scripts also read previous overlap statistics. Requires scipy/statsmodels for statistics. Check matched replicate completeness and hardcoded root paths.
