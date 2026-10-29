# Source-to-working-copy migration manifest

This is a **review-stage** reorganization. Each archive file is byte-identical to the file at the original `main` audit snapshot; active copies are also byte-identical. No script logic or paths have been edited. No analysis execution has been validated yet.

Original `main` branch: [audit snapshot](https://github.com/jpdols/calvados-elp-code-audit/tree/main).

**165 archived source files** (excluding the root Git configuration), **82 classified working copies**. All source files remain available under `archive/original_snapshot/`.

| Original source path (mirrored under `archive/original_snapshot/`) | Working destination / disposition |
|---|---|
| `analysis_library_1/plot_arch_reps_summary.py` | `supplementary/early_library/plot_arch_reps_summary.py` |
| `analysis_library_1/plot_calvados_architecture_summary.py` | Archive only — review if needed |
| `analysis_library_1/plot_calvados_architecture_summary_v2.py` | `supplementary/early_library/plot_calvados_architecture_summary_v2.py` |
| `analysis_library_2/900/contact_heatmaps_by_state.py` | `utilities/trajectory_processing/contact_heatmaps_by_state.py` |
| `analysis_library_2/900/idr.fasta` | Archive only — review if needed |
| `analysis_library_2/plot_calvados_architecture_summary_v2.py` | Archive only — review if needed |
| `analysis_library_2/plot_vgigip_replicate_landscapes.py` | `supplementary/early_library/plot_vgigip_replicate_landscapes.py` |
| `analysis_library_2/triblock/contact_heatmaps_by_state.py` | Archive only — review if needed |
| `analysis_library_2/triblock/idr.fasta` | Archive only — review if needed |
| `data/architecture_column_to_manifest.py` | Archive only — review if needed |
| `data/plot_library_from_summary.py` | `supplementary/early_library/plot_library_from_summary.py` |
| `data/scale_lambdas.py` | Archive only — review if needed |
| `final_analysis_pipeline/analyse_dwell_distributions.py` | `utilities/rg_kinetics/analyse_dwell_distributions.py` |
| `final_analysis_pipeline/analyse_rg_replicates_v2.py` | `utilities/rg_kinetics/analyse_rg_replicates_v2.py` |
| `final_analysis_pipeline/analyse_rg_single_library_fixed.py` | `utilities/rg_kinetics/analyse_rg_single_library_fixed.py` |
| `final_analysis_pipeline/investigate_vgipkg_properties.py` | `utilities/property_exploration/investigate_vgipkg_properties.py` |
| `final_analysis_pipeline/plot_architecture_threshold_independent.py` | `utilities/rg_kinetics/plot_architecture_threshold_independent.py` |
| `final_analysis_pipeline/plot_mean_rg_architecture_comparison.py` | `utilities/rg_kinetics/plot_mean_rg_architecture_comparison.py` |
| `final_analysis_pipeline/plot_rg_architecture_time.py` | `utilities/rg_kinetics/plot_rg_architecture_time.py` |
| `final_analysis_pipeline/plot_rg_kinetics.py` | `utilities/rg_kinetics/plot_rg_kinetics.py` |
| `final_analysis_pipeline/plot_vgipkg_overview_replicates.py` | `utilities/property_exploration/plot_vgipkg_overview_replicates.py` |
| `final_analysis_pipeline/plot_vgipkg_selected_370K_properties.py` | `utilities/property_exploration/plot_vgipkg_selected_370K_properties.py` |
| `plotting_thesis/Design_rule_1/plot_conformational_regimes.py` | Archive only — review if needed |
| `plotting_thesis/Design_rule_1/plot_conformational_regimes_DR1ab_v2_LARGE_TEXT.py` | Archive only — review if needed |
| `plotting_thesis/Design_rule_1/plot_conformational_regimes_DR1ab_v3_FIXED_ARCH_NAMES.py` | Archive only — review if needed |
| `plotting_thesis/Design_rule_1/plot_conformational_regimes_DR1ab_v4_COMPOSITION_BOXES.py` | Archive only — review if needed |
| `plotting_thesis/Design_rule_1/plot_conformational_regimes_DR1ab_v5_FULL_RANGE_COLORED.py` | Archive only — review if needed |
| `plotting_thesis/Design_rule_1/plot_conformational_regimes_DR1ab_v6_SOLID_MEANS.py` | Archive only — review if needed |
| `plotting_thesis/Design_rule_1/plot_conformational_regimes_DR1ab_v7_300K_SUPPLEMENT.py` | `design_rules/DR01/supplementary/plot_conformational_regimes_DR1ab_v7_300K_SUPPLEMENT.py` |
| `plotting_thesis/Design_rule_1/plot_conformational_regimes_updated.py` | `design_rules/DR01/plotting/plot_conformational_regimes_updated.py` |
| `plotting_thesis/analyse_triplicate_statistics.py` | `supplementary/cross_rule/analyse_triplicate_statistics.py` |
| `plotting_thesis/design_rule_2/analyse_rg_distribution_overlap.py` | `design_rules/DR02/analysis/analyse_rg_distribution_overlap.py` |
| `plotting_thesis/design_rule_2/analyse_triplicate_statistics_12pairs.py` | `design_rules/DR02/analysis/analyse_triplicate_statistics_12pairs.py` |
| `plotting_thesis/design_rule_2/plot_DR2_conformational_regime.py` | `design_rules/DR02/supplementary/plot_DR2_conformational_regime.py` |
| `plotting_thesis/design_rule_2/plot_DR2_thesis_final_v10_COMPACT_SPACING_TALL_AXES.py` | Archive only — review if needed |
| `plotting_thesis/design_rule_2/plot_DR2_thesis_final_v11_BOLD_AXIS_LABELS.py` | Archive only — review if needed |
| `plotting_thesis/design_rule_2/plot_DR2_thesis_final_v12_BOLD_TICK_DETAILS.py` | Archive only — review if needed |
| `plotting_thesis/design_rule_2/plot_DR2_thesis_final_v13_LARGE_LABELS.py` | Archive only — review if needed |
| `plotting_thesis/design_rule_2/plot_DR2_thesis_final_v14_MUTED_GREEN.py` | `design_rules/DR02/plotting/plot_DR2_thesis_final_v14_MUTED_GREEN.py` |
| `plotting_thesis/design_rule_2/plot_design_rule_2_architecture_contrast.py` | `design_rules/DR02/supplementary/plot_design_rule_2_architecture_contrast.py` |
| `plotting_thesis/design_rule_2/supplementary.py` | `design_rules/DR02/supplementary/supplementary.py` |
| `plotting_thesis/design_rule_3/architecture_DR3_main_and_supplement_v6.py` | `design_rules/DR03/supplementary/architecture_DR3_main_and_supplement_v6.py` |
| `plotting_thesis/design_rule_3/motif_contacts/plot_DR3_BB_all_vs_interpatch.py` | `design_rules/DR03/supplementary/motif_contacts/plot_DR3_BB_all_vs_interpatch.py` |
| `plotting_thesis/design_rule_3/motif_contacts/plot_DR3_BB_two_panel_thesis_v2.py` | `design_rules/DR03/supplementary/motif_contacts/plot_DR3_BB_two_panel_thesis_v2.py` |
| `plotting_thesis/design_rule_3/motif_contacts/plot_DR3_R4_BB_separation_options.py` | `design_rules/DR03/supplementary/motif_contacts/plot_DR3_R4_BB_separation_options.py` |
| `plotting_thesis/design_rule_3/motif_contacts/plot_DR3_motif_contacts_final_v7_thesis.py` | Archive only — review if needed |
| `plotting_thesis/design_rule_3/motif_contacts/plot_DR3_motif_contacts_final_v8_thesis.py` | `design_rules/DR03/supplementary/motif_contacts/plot_DR3_motif_contacts_final_v8_thesis.py` |
| `plotting_thesis/design_rule_3/motif_contacts/plot_DR3_nonlocal_BB_contact_options.py` | `design_rules/DR03/supplementary/motif_contacts/plot_DR3_nonlocal_BB_contact_options.py` |
| `plotting_thesis/design_rule_3/plot_DR3_three_thesis_figures.py` | Archive only — review if needed |
| `plotting_thesis/design_rule_3/plot_DR3_three_thesis_figures_v10.py` | Archive only — review if needed |
| `plotting_thesis/design_rule_3/plot_DR3_three_thesis_figures_v11.py` | Archive only — review if needed |
| `plotting_thesis/design_rule_3/plot_DR3_three_thesis_figures_v12.py` | Archive only — review if needed |
| `plotting_thesis/design_rule_3/plot_DR3_three_thesis_figures_v13.py` | `design_rules/DR03/plotting/plot_DR3_three_thesis_figures_v13.py` |
| `plotting_thesis/design_rule_3/plot_DR3_three_thesis_figures_v2.py` | Archive only — review if needed |
| `plotting_thesis/design_rule_3/plot_DR3_three_thesis_figures_v3.py` | Archive only — review if needed |
| `plotting_thesis/design_rule_3/plot_DR3_three_thesis_figures_v5.py` | Archive only — review if needed |
| `plotting_thesis/design_rule_3/plot_DR3_three_thesis_figures_v6.py` | Archive only — review if needed |
| `plotting_thesis/design_rule_3/plot_DR3_three_thesis_figures_v8.py` | Archive only — review if needed |
| `plotting_thesis/design_rule_3/plot_DR3_three_thesis_figures_v9.py` | Archive only — review if needed |
| `plotting_thesis/design_rule_3/plot_architecture_design_rules.py` | `design_rules/DR03/supplementary/plot_architecture_design_rules.py` |
| `plotting_thesis/design_rule_3/plot_architecture_thesis_figures.py` | `design_rules/DR03/supplementary/plot_architecture_thesis_figures.py` |
| `plotting_thesis/design_rule_3/plot_architecture_thesis_figures_v3_R4_independent.py` | Archive only — review if needed |
| `plotting_thesis/design_rule_3/plot_architecture_thesis_figures_v4_R4_from_all_rg.py` | Archive only — review if needed |
| `plotting_thesis/design_rule_3/plot_architecture_thesis_figures_v5_random_first.py` | Archive only — review if needed |
| `plotting_thesis/design_rule_3/plot_within_ratio_heatmaps.py` | `design_rules/DR03/supplementary/plot_within_ratio_heatmaps.py` |
| `plotting_thesis/design_rule_4/analyse_charge_distribution_rg_v3.py` | `design_rules/DR04/analysis/analyse_charge_distribution_rg_v3.py` |
| `plotting_thesis/design_rule_4/plot_design_rule4_architecture_average_v6.py` | `design_rules/DR04/supplementary/plot_design_rule4_architecture_average_v6.py` |
| `plotting_thesis/design_rule_4/plot_design_rule4_regime_overview_v2.py` | `design_rules/DR04/supplementary/plot_design_rule4_regime_overview_v2.py` |
| `plotting_thesis/design_rule_4/plot_design_rule4_thesis_v11.py` | Archive only — review if needed |
| `plotting_thesis/design_rule_4/plot_design_rule4_thesis_v17.py` | Archive only — review if needed |
| `plotting_thesis/design_rule_4/plot_design_rule4_thesis_v18.py` | Archive only — review if needed |
| `plotting_thesis/design_rule_4/plot_design_rule4_thesis_v19_print.py` | Archive only — review if needed |
| `plotting_thesis/design_rule_4/plot_design_rule4_thesis_v20_final.py` | `design_rules/DR04/plotting/plot_design_rule4_thesis_v20_final.py` |
| `plotting_thesis/design_rule_4/plot_design_rule4_thesis_v5.py` | Archive only — review if needed |
| `plotting_thesis/design_rule_4/plot_design_rule4_thesis_v6.py` | Archive only — review if needed |
| `plotting_thesis/design_rule_4/plot_design_rule4_thesis_v7.py` | Archive only — review if needed |
| `plotting_thesis/design_rule_4/plot_design_rule4_thesis_v8.py` | Archive only — review if needed |
| `plotting_thesis/design_rule_4/plot_design_rule4_thesis_v9.py` | Archive only — review if needed |
| `plotting_thesis/design_rule_5/analyse_dr5_rg.py` | `design_rules/DR05/analysis/analyse_dr5_rg.py` |
| `plotting_thesis/design_rule_5/analyse_dr5_ternary_rg.py` | `design_rules/DR05/analysis/analyse_dr5_ternary_rg.py` |
| `plotting_thesis/design_rule_5/plot_dr5_scd_370K_FIXED.py` | Archive only — review if needed |
| `plotting_thesis/design_rule_5/plot_dr5_scd_370K_LABELS_FIXED.py` | Archive only — review if needed |
| `plotting_thesis/design_rule_5/plot_dr5_scd_370K_PRINT_LARGE.py` | Archive only — review if needed |
| `plotting_thesis/design_rule_5/plot_dr5_scd_370K_PRINT_REDESIGNED.py` | `design_rules/DR05/plotting/plot_dr5_scd_370K_PRINT_REDESIGNED.py` |
| `plotting_thesis/design_rule_5/plot_dr5_scd_labels.py` | Archive only — review if needed |
| `plotting_thesis/design_rule_5/plot_dr5_ternary_comparisons.py` | `design_rules/DR05/supplementary/plot_dr5_ternary_comparisons.py` |
| `plotting_thesis/design_rule_5/plot_dr5_ternary_deviations.py` | `design_rules/DR05/supplementary/plot_dr5_ternary_deviations.py` |
| `plotting_thesis/design_rule_5/plot_dr5_ternary_focused.py` | Archive only — review if needed |
| `plotting_thesis/design_rule_5/plot_dr5_ternary_focused_updated.py` | `design_rules/DR05/supplementary/plot_dr5_ternary_focused_updated.py` |
| `plotting_thesis/design_rule_6/analyse_dr6_vgigkg_rg.py` | `design_rules/DR06/analysis/analyse_dr6_vgigkg_rg.py` |
| `plotting_thesis/design_rule_6/plot_dr6_architecture_focus_v11_large_fonts.py` | Archive only — review if needed |
| `plotting_thesis/design_rule_6/plot_dr6_architecture_focus_v12_readable_balanced.py` | `design_rules/DR06/plotting/plot_dr6_architecture_focus_v12_readable_balanced.py` |
| `plotting_thesis/design_rule_6/plot_dr6_architecture_focus_v3.py` | Archive only — review if needed |
| `plotting_thesis/design_rule_6/plot_dr6_architecture_focus_v5.py` | Archive only — review if needed |
| `plotting_thesis/design_rule_6/plot_dr6_architecture_focus_v6.py` | Archive only — review if needed |
| `plotting_thesis/design_rule_6/plot_dr6_four_panels.py` | Archive only — review if needed |
| `plotting_thesis/design_rule_6/plot_dr6_full_library_all_ratios.py` | `design_rules/DR06/supplementary/plot_dr6_full_library_all_ratios.py` |
| `plotting_thesis/design_rule_6/plot_dr6_full_library_all_ratios_pink.py` | Archive only — review if needed |
| `plotting_thesis/design_rule_6/plot_dr6_library_design_overview.py` | `design_rules/DR06/supplementary/plot_dr6_library_design_overview.py` |
| `plotting_thesis/design_rule_6/plot_dr6_ratio_split.py` | Archive only — review if needed |
| `plotting_thesis/design_rule_6/plot_dr6_ratio_split_extended.py` | `design_rules/DR06/supplementary/plot_dr6_ratio_split_extended.py` |
| `plotting_thesis/design_rule_6/plot_dr6_ternary_focused.py` | `design_rules/DR06/supplementary/plot_dr6_ternary_focused.py` |
| `plotting_thesis/design_rule_6/plot_dr6_vgigkg_comparisons.py` | Archive only — review if needed |
| `plotting_thesis/design_rule_6/plot_dr6_vgigkg_comparisons_updated.py` | `design_rules/DR06/supplementary/plot_dr6_vgigkg_comparisons_updated.py` |
| `plotting_thesis/design_rule_6/plot_dr6a_all_ratios.py` | `design_rules/DR06/supplementary/plot_dr6a_all_ratios.py` |
| `plotting_thesis/design_rule_7/plot_dr7_overview_3rep.py` | `design_rules/DR07/plotting/plot_dr7_overview_3rep.py` |
| `plotting_thesis/design_rule_7/plot_dr7_thesis.py` | `design_rules/DR07/plotting/plot_dr7_thesis.py` |
| `plotting_thesis/design_rule_8/create_dr8_complete_summary.py` | `design_rules/DR08/analysis/create_dr8_complete_summary.py` |
| `plotting_thesis/design_rule_8/create_vgipkg_summary_kinetics.py` | `design_rules/DR08/analysis/create_vgipkg_summary_kinetics.py` |
| `plotting_thesis/design_rule_8/plot_dr8_complete_story.py` | `design_rules/DR08/plotting/plot_dr8_complete_story.py` |
| `plotting_thesis/design_rule_8/plot_dr8_thesis.py` | Archive only — review if needed |
| `plotting_thesis/design_rule_8/plot_dr8_thesis_updated.py` | `design_rules/DR08/plotting/plot_dr8_thesis_updated.py` |
| `plotting_thesis/design_rule_8/plot_dr8_three_subrules.py` | `design_rules/DR08/plotting/plot_dr8_three_subrules.py` |
| `plotting_thesis/design_rule_8/plot_dr8_vgipkg_kinetics.py` | `design_rules/DR08/plotting/plot_dr8_vgipkg_kinetics.py` |
| `plotting_thesis/methods/make_methods_supplement_architectures.py` | Archive only — review if needed |
| `plotting_thesis/methods/make_methods_supplement_architectures_thesis.py` | `supplementary/methods/make_methods_supplement_architectures_thesis.py` |
| `plotting_thesis/methods/plot_central90_explanation.py` | `supplementary/methods/plot_central90_explanation.py` |
| `plotting_thesis/methods/plot_motif_architectures.py` | Archive only — review if needed |
| `plotting_thesis/methods/plot_motif_architectures_thesis.py` | `supplementary/methods/plot_motif_architectures_thesis.py` |
| `plotting_thesis/model_references/lab/analyse_lab_temperature_sweep.py` | Archive only — review if needed |
| `plotting_thesis/model_references/lab/analyse_lab_temperature_sweep_v2_thesis.py` | Archive only — review if needed |
| `plotting_thesis/model_references/lab/analyse_lab_temperature_sweep_v3_all_sequences.py` | Archive only — review if needed |
| `plotting_thesis/model_references/lab/analyse_lab_temperature_sweep_v4_recursive_sequence_search.py` | Archive only — review if needed |
| `plotting_thesis/model_references/lab/analyse_lab_temperature_sweep_v5_EAI_two_layouts.py` | Archive only — review if needed |
| `plotting_thesis/model_references/lab/analyse_lab_temperature_sweep_v6_FINAL_C_inset.py` | `model_validation/lab/analyse_lab_temperature_sweep_v6_FINAL_C_inset.py` |
| `plotting_thesis/model_references/lcst/analyse_reference_architecture_370K_v2.py` | `model_validation/reference_architectures/analyse_reference_architecture_370K_v2.py` |
| `plotting_thesis/model_references/paper/analyse_reference_temperature_sweeps.py` | `model_validation/paper/analyse_reference_temperature_sweeps.py` |
| `plotting_thesis/model_references/references/plot_base_elp_references.py` | `model_validation/reference_elp/plot_base_elp_references.py` |
| `plotting_thesis/model_references/references/plot_urry_correlation.py` | `model_validation/reference_elp/plot_urry_correlation.py` |
| `plotting_thesis/scaling/analyse_gamma_scaling.py` | `model_validation/scaling/analyse_gamma_scaling.py` |
| `plotting_thesis/scaling/plot_gamma_scaling_thesis.py` | `model_validation/scaling/plot_gamma_scaling_thesis.py` |
| `plotting_thesis/scaling/plot_lambda_gamma.py` | `model_validation/scaling/plot_lambda_gamma.py` |
| `plotting_thesis/standard_rg_analysis.py` | `utilities/rg_kinetics/standard_rg_analysis.py` |
| `runs/check_pbc_splits.py` | `utilities/trajectory_processing/check_pbc_splits.py` |
| `runs/plot_rg.py` | Archive only — review if needed |
| `runs/plot_rg_multi.py` | `utilities/trajectory_processing/plot_rg_multi.py` |
| `runs/scale1/E_AI_library/analyse_arch_replicates.py` | `utilities/diagnostics/analyse_arch_replicates.py` |
| `runs/scale1/E_AI_library/analyse_contact_ranges.py` | `utilities/diagnostics/analyse_contact_ranges.py` |
| `runs/scale1/E_AI_library/build_E_AI_370_library_90runs.py` | Archive only — review if needed |
| `runs/scale1/E_AI_library/plot_block_hydrophobicity_barcodes.py` | Archive only — review if needed |
| `runs/scale1/E_AI_library/plot_eai_arch_summary.py` | Archive only — review if needed |
| `runs/scale1/E_AI_library/plot_lambda_chain.py` | Archive only — review if needed |
| `runs/scale1/E_AI_library/plot_scrambled_hydrophobicity_barcodes.py` | Archive only — review if needed |
| `runs/scale1/E_AI_library/prepare.py` | Archive only — review if needed |
| `runs/scale1/E_AI_library/run_1us.slurm` | Archive only — review if needed |
| `runs/scale1/E_AI_library/submit_all.sh` | Archive only — review if needed |
| `runs/scale1/analyse_shape_dynamics.py` | `utilities/trajectory_processing/analyse_shape_dynamics.py` |
| `runs/scale1/markov_rg_theta_analysis.py` | `utilities/diagnostics/markov_rg_theta_analysis.py` |
| `runs/scale1/markov_validation_tests.py` | `utilities/diagnostics/markov_validation_tests.py` |
| `runs/scale1/plot_arch_reps_summary.py` | Archive only — review if needed |
| `runs/scale1/plot_overview.py` | Archive only — review if needed |
| `runs/scale1/plot_rg_ecdf_arch_compare.py` | `utilities/trajectory_processing/plot_rg_ecdf_arch_compare.py` |
| `src/calvados_lambda_ref.py` | Archive only — review if needed |
| `src/inspect_tea_tables.py` | Archive only — review if needed |
| `src/main.py` | Archive only — review if needed |
| `src/make_calvados_residue_file.py` | Archive only — review if needed |
| `src/make_tea_tables.py` | Archive only — review if needed |
| `src/tea_hydration_data.py` | Archive only — review if needed |
| `src/tea_kappa_data.py` | Archive only — review if needed |
| `src/tea_lambda.py` | Archive only — review if needed |
| `src/tea_thermo.py` | Archive only — review if needed |
| `tests/E_AI_triblock/plot_rg_multi.py` | Archive only — review if needed |
| `tests/plot_I_contacts.py` | Archive only — review if needed |
| `tests/plot_hist_contactnumber.py` | `utilities/diagnostics/plot_hist_contactnumber.py` |
| `tests/plot_rg_multi.py` | `utilities/diagnostics/plot_rg_multi.py` |

## Outstanding needs

- Recover DelftBlue's `analyse_library_overview.py` and trajectory/contact preprocessing source code missing from this snapshot.
- Match candidate final figure scripts to the submitted thesis and verify the exact input CSV versions.
- Record temperature, stride, frame interval, collapse threshold, state definition, replicate mapping, FASTA sequences and input data checksums.
- Do not treat archived or working code as independently reproducible until a test run has passed.
