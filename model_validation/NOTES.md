# Model validation — proposed scripts

Candidate sources from `plotting_thesis/`:
- `model_references/lab/analyse_lab_temperature_sweep_v6_FINAL_C_inset.py` — lab ELP temperature sweeps. Inputs: top.pdb, DCD trajectories, idr.fasta (or supplied summary CSV).
- `model_references/paper/analyse_reference_temperature_sweeps.py` — paper reference temperature sweeps. Inputs: DCD/PDB/FASTA.
- `model_references/lcst/analyse_reference_architecture_370K_v2.py` — E_AI / E_AI_triblock / E_AE / E_AE_triblock, raw DCD/PDB.
- `model_references/references/plot_base_elp_references.py` — base ELP reference response, raw DCD/PDB; hardcoded paths to fix.
- `model_references/references/plot_urry_correlation.py` — Urry vs CALVADOS lambda correlation; inputs: temperature-specific residue tables / merged reference data.
- `scaling/analyse_gamma_scaling.py` — gamma scaling from trajectories and residue tables.
- `scaling/plot_gamma_scaling_thesis.py` — reads `gamma_scaling_summary.csv`.
- `scaling/plot_lambda_gamma.py` — reads temperature-specific residue parameter CSVs.

Earlier lab variants v1–v5: archive/review, not automatically omit. Confirm final thesis usage. Note that several scripts require full trajectories; do not claim they run from `all_rg_values.csv` alone.
