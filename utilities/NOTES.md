# General analysis utilities — candidate scripts

**Trajectory processing (needs DCD/PDB):** `analyse_library_overview.py` (missing; recover from DelftBlue), `analysis_library_2/900/contact_heatmaps_by_state.py` (identical to triblock copy), `runs/scale1/analyse_shape_dynamics.py`, `runs/check_pbc_splits.py`, `runs/plot_rg_multi.py`, `runs/scale1/plot_rg_ecdf_arch_compare.py`.

**CSV-based Rg/kinetics:** `plotting_thesis/standard_rg_analysis.py`, `final_analysis_pipeline/analyse_rg_single_library_fixed.py`, `analyse_rg_replicates_v2.py`, `analyse_dwell_distributions.py`, `plot_rg_architecture_time.py`, `plot_rg_kinetics.py`, `plot_mean_rg_architecture_comparison.py`, `plot_architecture_threshold_independent.py`.

**Property exploration:** `final_analysis_pipeline/investigate_vgipkg_properties.py`, `plot_vgipkg_selected_370K_properties.py`, `plot_vgipkg_overview_replicates.py`.

**Early architecture/contacts diagnostics:** `runs/scale1/markov_rg_theta_analysis.py`, `markov_validation_tests.py`, `runs/scale1/E_AI_library/analyse_arch_replicates.py`, `analyse_contact_ranges.py`, `tests/plot_hist_contactnumber.py`, `tests/plot_rg_multi.py`.

**Important:** state definitions differ. `standard_rg_analysis.py` uses the saved `collapsed` column; other programs derive threshold/hysteresis states. Retain per-script semantics; do not silently homogenize metrics.
