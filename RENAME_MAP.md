# Rename Map

This map records public repository file/module renames from the 2026-09-14
repository restructure. Historical artifact paths under `outputs/` are not
rewritten.

## Script Modules

| Old module or file | New module or file | Status | Notes |
|---|---|---|---|
| analyze_m1m2_paid_pilot | analyze_evidence_selection_pilot | RENAMED | Evidence-selection pilot analysis. |
| freeze_m1m2_paid_pilot_manifest | freeze_evidence_selection_pilot_manifest | RENAMED | Evidence-selection pilot manifest freezer. |
| run_m1m2_paid_pilot | run_evidence_selection_pilot | RENAMED | Evidence-selection pilot runner. |
| analyze_m1m2_selective_paid_pilot | analyze_evidence_selection_selective_propagation_pilot | RENAMED | Selective-propagation pilot analysis. |
| freeze_m1m2_selective_paid_pilot_manifest | freeze_evidence_selection_selective_propagation_pilot_manifest | RENAMED | Selective-propagation pilot manifest freezer. |
| run_m1m2_selective_paid_pilot | run_evidence_selection_selective_propagation_pilot | RENAMED | Selective-propagation pilot runner. |
| analyze_m1m2_selective_confirmation | analyze_evidence_selection_selective_propagation_confirmation | RENAMED | Selective-propagation confirmation analysis. |
| freeze_m1m2_selective_confirmation_manifest | freeze_evidence_selection_selective_propagation_confirmation_manifest | RENAMED | Selective-propagation confirmation manifest freezer. |
| audit_m1m2_selective_confirmation_pre_execution | audit_evidence_selection_selective_propagation_pre_execution | RENAMED | Selective-propagation pre-execution audit. |
| run_m1m2_selective_confirmation | run_evidence_selection_selective_propagation_confirmation | RENAMED | Selective-propagation confirmation runner. |
| build_m1m2_selective_stress_roster | build_evidence_selection_selective_propagation_robustness_roster | RENAMED | Robustness roster for selective propagation. |
| project_m1m2_selective_zero_cost | project_evidence_selection_selective_propagation_zero_cost | RENAMED | Zero-cost selective-propagation projection. |
| analyze_m1m2_stress_paid_pilot | analyze_robustness_pilot | RENAMED | Robustness pilot analysis. |
| freeze_m1m2_stress_paid_pilot_manifest | freeze_robustness_pilot_manifest | RENAMED | Robustness pilot manifest freezer. |
| run_m1m2_stress_paid_pilot | run_robustness_pilot | RENAMED | Robustness pilot runner. |
| build_m1m2_stress_benchmark | build_robustness_benchmark | RENAMED | Robustness benchmark builder. |
| audit_m1m2_stress_128_source_pool | audit_robustness_source_pool | RENAMED | Robustness source-pool audit. |
| freeze_m1m2_stress_128_manifest | freeze_robustness_manifest | RENAMED | Robustness confirmation manifest freezer. |
| run_m1m2_stress_128_confirmation | run_robustness_confirmation | RENAMED | Robustness confirmation runner. |
| audit_m1m2_candidate_guided_postrun_governance | audit_evidence_selection_candidate_guided_postrun_governance | RENAMED | Candidate-guided postrun governance audit. |
| build_m1m2_candidate_guided_roster | build_evidence_selection_candidate_guided_roster | RENAMED | Candidate-guided roster builder. |
| freeze_m1m2_candidate_guided_confirmation_manifest | freeze_evidence_selection_candidate_guided_confirmation_manifest | RENAMED | Candidate-guided confirmation manifest freezer. |
| run_m1m2_candidate_guided_confirmation | run_evidence_selection_candidate_guided_confirmation | RENAMED | Candidate-guided confirmation runner. |
| build_m1m2_answerability_preserving_roster | build_evidence_selection_answerability_roster | RENAMED | Answerability-preserving roster builder. |
| project_m1m2_answerability_preserving_zero_cost | project_evidence_selection_answerability_zero_cost | RENAMED | Answerability-preserving zero-cost projection. |
| project_m1m2_guarded_zero_cost | project_evidence_selection_guarded_zero_cost | RENAMED | Guarded zero-cost projection. |
| build_m1m2_schema_outputs | build_evidence_selection_schema_outputs | RENAMED | Evidence-selection schema output builder. |
| audit_m1m2_plotqa_guardrail | audit_evidence_selection_plotqa_guardrail | RENAMED | PlotQA guardrail audit for evidence selection. |
| audit_m1m2_recovery_progress | audit_evidence_selection_recovery | RENAMED | Evidence-selection recovery progress audit. |
| analyze_track_b_m1m2_identifiability | analyze_evidence_selection_identifiability | RENAMED | `progress.md` may contain old narrative references. |
| prepare_m1m2_recovery_phase_b | prepare_evidence_selection_preregistration | RENAMED | Evidence-selection preregistration preparation. |
| run_m1m2_recovery_phase_a | run_evidence_selection_variance_study | RENAMED | Variance-study runner. |
| run_m1m2_recovery_phase_c_pilot | run_quality_gradient_pilot | RENAMED | Quality-gradient pilot runner. |
| prepare_m1m2_recovery_round3 | prepare_recovery_round | RENAMED | Numbered recovery round replaced with semantic name. |
| analyze_p0_4_extrema_validation | analyze_extrema_validation | RENAMED | Extrema validation analysis. |
| run_p0_3_extrema_validation | run_extrema_validation | RENAMED | Extrema validation runner. |
| run_p1_3_prompt_pilot | run_prompt_fairness_pilot | RENAMED | Prompt fairness pilot runner. |
| estimate_p1_4_rerun_budget | estimate_rerun_budget | RENAMED | Rerun budget estimator. |
| analyze_plotqa_stage5_projection | analyze_plotqa_repair_projection | RENAMED | PlotQA repair projection analysis. |
| run_plotqa_measured_geosym_stage3 | run_plotqa_locate | RENAMED | PlotQA locate-stage runner. |
| run_plotqa_measured_geosym_stage4 | run_plotqa_measure | RENAMED | PlotQA measure-stage runner. |
| run_plotqa_measured_geosym_stage5_repair_smoke | run_plotqa_repair_smoke | RENAMED | PlotQA repair smoke runner. |
| run_plotqa_direct_public_500_baselines | run_plotqa_direct_public_baselines | RENAMED | Public PlotQA direct-baseline runner. |
| run_plotqa_measured_public_500_confirmation | run_plotqa_measured_public_confirmation | RENAMED | Public PlotQA measured-data confirmation runner. |
| m1m2_recovery_quality_gradient | m1m2_recovery_quality_gradient | PROVENANCE-LOCKED | `preregistration.json` records `scripts.m1m2_recovery_quality_gradient.degrade_coords`; renaming would break the frozen preregistration pointer. |
| run_plotqa_measured_locate_smoke | run_plotqa_measured_locate_smoke | PROVENANCE-LOCKED | Referenced by frozen `repair_history.json`; already semantic enough for file-name policy. |
| run_plotqa_measured_m0 | run_plotqa_measured_m0 | PROVENANCE-LOCKED | Referenced by frozen `repair_history.json`; already semantic enough for file-name policy. |
| selective_propagation | selective_propagation | NO-CHANGE-NEEDED | Already semantic and explicitly exempted by the restructure plan. |
| prepare_m1m2_recovery_round4 | freeze_recovery_repair_round | RENAMED | Later bounded repair-round manifest freezer; kept tracked because it freezes the paper-referenced 18/24 clean-stop round. |

## Test Files

| Old file | New file | Status | Notes |
|---|---|---|---|
| tests/test_m1m2_paid_pilot_analysis.py | tests/test_evidence_selection_pilot_analysis.py | RENAMED | Mirrors script rename. |
| tests/test_m1m2_paid_pilot_manifest.py | tests/test_evidence_selection_pilot_manifest.py | RENAMED | Mirrors script rename. |
| tests/test_m1m2_paid_pilot_runner.py | tests/test_evidence_selection_pilot_runner.py | RENAMED | Mirrors script rename. |
| tests/test_m1m2_selective_paid_pilot_analysis.py | tests/test_evidence_selection_selective_propagation_pilot_analysis.py | RENAMED | Mirrors script rename. |
| tests/test_m1m2_selective_paid_pilot_runner.py | tests/test_evidence_selection_selective_propagation_pilot_runner.py | RENAMED | Mirrors script rename. |
| tests/test_m1m2_selective_confirmation_analysis.py | tests/test_evidence_selection_selective_propagation_confirmation_analysis.py | RENAMED | Mirrors script rename. |
| tests/test_m1m2_selective_confirmation_manifest.py | tests/test_evidence_selection_selective_propagation_confirmation_manifest.py | RENAMED | Mirrors script rename. |
| tests/test_m1m2_selective_pre_execution_audit.py | tests/test_evidence_selection_selective_propagation_pre_execution_audit.py | RENAMED | Mirrors script rename. |
| tests/test_m1m2_selective_confirmation_runner.py | tests/test_evidence_selection_selective_propagation_confirmation_runner.py | RENAMED | Mirrors script rename. |
| tests/test_m1m2_selective_stress_roster.py | tests/test_evidence_selection_selective_propagation_robustness_roster.py | RENAMED | Mirrors script rename. |
| tests/test_m1m2_selective_zero_cost_and_manifest.py | tests/test_evidence_selection_selective_propagation_zero_cost_and_manifest.py | RENAMED | Mirrors script rename. |
| tests/test_m1m2_stress_paid_pilot_analysis.py | tests/test_robustness_pilot_analysis.py | RENAMED | Mirrors script rename. |
| tests/test_m1m2_stress_paid_pilot_manifest.py | tests/test_robustness_pilot_manifest.py | RENAMED | Mirrors script rename. |
| tests/test_m1m2_stress_paid_pilot_runner.py | tests/test_robustness_pilot_runner.py | RENAMED | Mirrors script rename. |
| tests/test_m1m2_stress_benchmark.py | tests/test_robustness_benchmark.py | RENAMED | Mirrors script rename. |
| tests/test_m1m2_stress_128_source_pool.py | tests/test_robustness_source_pool.py | RENAMED | Mirrors script rename. |
| tests/test_m1m2_stress_128_confirmation_runner.py | tests/test_robustness_confirmation_runner.py | RENAMED | Mirrors script rename. |
| tests/test_m1m2_candidate_guided_postrun_governance.py | tests/test_evidence_selection_candidate_guided_postrun_governance.py | RENAMED | Mirrors script rename. |
| tests/test_m1m2_candidate_guided_confirmation_runner.py | tests/test_evidence_selection_candidate_guided_confirmation_runner.py | RENAMED | Mirrors script rename. |
| tests/test_m1m2_candidate_guided_protocol.py | tests/test_evidence_selection_candidate_guided_protocol.py | RENAMED | Mirrors script rename. |
| tests/test_m1m2_answerability_preserving_zero_cost.py | tests/test_evidence_selection_answerability_zero_cost.py | RENAMED | Mirrors script rename. |
| tests/test_m1m2_guarded_projection.py | tests/test_evidence_selection_guarded_projection.py | RENAMED | Mirrors script rename. |
| tests/test_m1m2_schema_outputs.py | tests/test_evidence_selection_schema_outputs.py | RENAMED | Mirrors script rename. |
| tests/test_m1m2_plotqa_guardrail_audit.py | tests/test_evidence_selection_plotqa_guardrail_audit.py | RENAMED | Mirrors script rename. |
| tests/test_m1m2_recovery_progress_audit.py | tests/test_evidence_selection_recovery_audit.py | RENAMED | Mirrors script rename. |
| tests/test_m1m2_recovery_phase_a.py | tests/test_evidence_selection_variance_study.py | RENAMED | Mirrors script rename. |
| tests/test_m1m2_recovery_phase_c_pilot.py | tests/test_quality_gradient_pilot.py | RENAMED | Mirrors script rename. |
| tests/test_m1m2_recovery_quality_gradient.py | tests/test_quality_gradient.py | RENAMED | Test file renamed while the provenance-locked module remains unchanged. |
| tests/test_p0_4_extrema_analysis.py | tests/test_extrema_validation_analysis.py | RENAMED | Mirrors script rename. |
| tests/test_p0_3_validation_script.py | tests/test_extrema_validation_runner.py | RENAMED | Mirrors script rename. |
| tests/test_p1_3_prompt_pilot.py | tests/test_prompt_fairness_pilot.py | RENAMED | Mirrors script rename. |
| tests/test_p1_4_budget.py | tests/test_rerun_budget.py | RENAMED | Mirrors script rename. |
| tests/test_plotqa_stage5_projection.py | tests/test_plotqa_repair_projection.py | RENAMED | Mirrors script rename. |
| tests/test_plotqa_measured_geosym_stage3.py | tests/test_plotqa_locate.py | RENAMED | Mirrors script rename. |
| tests/test_plotqa_measured_geosym_stage4.py | tests/test_plotqa_measure.py | RENAMED | Mirrors script rename. |
| tests/test_plotqa_measured_geosym_stage5_repair_smoke.py | tests/test_plotqa_repair_smoke.py | RENAMED | Mirrors script rename. |
| tests/test_plotqa_direct_public_500_baselines.py | tests/test_plotqa_direct_public_baselines.py | RENAMED | Mirrors script rename. |
| tests/test_plotqa_measured_public_500_confirmation.py | tests/test_plotqa_measured_public_confirmation.py | RENAMED | Mirrors script rename. |
