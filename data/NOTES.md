# Data policy and missing inputs

This directory is currently a **placeholder**: the snapshot has almost no numerical data. Only lightweight FASTA examples are present in the audit.

Suggested Git-tracked data: definitive sequence manifests and FASTA, versioned summary CSV files actually read by final figures, small validation examples, metadata describing settings and units.

Large frame-resolved `all_rg_values.csv` and full DCD/PDB trajectories: publish in a separate durable archive and reference by URL/checksum. Never assume `per_run_summary.csv` can reconstruct time-ordered dwell events.

For each dataset record: study, sequence IDs, temperature, replicate, simulation version, timestep/frame interval, analysis stride, Rg unit, collapse threshold, state algorithm, censoring policy, source files, script commit, checksum.

Missing upstream outputs include `all_rg_values.csv`, `per_run_summary.csv`, `dwell_events.csv`, `rg_summary_all_libraries.csv`, `combined_summary_mean_std.csv`, `design_metadata.csv`, contact summary outputs, and precise input FASTA files.
