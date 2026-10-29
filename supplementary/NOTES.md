# Supplementary figures and cross-rule analyses

**Methods / architecture schematics:**
- `plotting_thesis/methods/plot_motif_architectures_thesis.py` (and base variant): produces R4 motif architecture diagrams, largely self-contained.
- `plotting_thesis/methods/make_methods_supplement_architectures_thesis.py` and base version: methods/supplement subset figures.
- `plotting_thesis/methods/plot_central90_explanation.py`: central 90% Rg range explanation; reads `all_rg_values.csv`, path cleanup needed.

**Cross-rule comparisons, statistics and exploratory materials:**
- `plotting_thesis/analyse_triplicate_statistics.py`: replicate statistics and multiple-comparison tests; reads `per_run_summary.csv`.
- `analysis_library_1/plot_calvados_architecture_summary_v2.py` (duplicate in `analysis_library_2/`): early library landscape figures from CSV outputs.
- `analysis_library_2/plot_vgigip_replicate_landscapes.py`: VGIGIP comparative landscapes; path cleanup.
- `data/plot_library_from_summary.py`: broad plots from `per_architecture_300_vs_370.csv`.

Place rule-specific supplementary figures inside their DR folder after review, not automatically in this top-level directory.
