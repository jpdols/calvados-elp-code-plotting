# Design Rule 1 — conformational regimes

**Candidate primary plot:** `plotting_thesis/Design_rule_1/plot_conformational_regimes_updated.py`.

Earlier variants: `plot_conformational_regimes.py`, `plot_conformational_regimes_DR1ab_v2_LARGE_TEXT.py` through `v7_300K_SUPPLEMENT.py`. Compare exact figure outputs with submitted thesis before selecting one.

**Inputs:** `all_rg_values.csv`; for some reference/baseline calculations `top.pdb` and `*.dcd`. Candidate outputs include `pure_baseline_statistics.csv`, `architecture_level_statistics.csv`, and figures.

**Review:** script has embedded path settings, has no regular CLI, and includes trajectory-specific calculations. Document baseline definitions, frame timing, and reproducible input locations.
