# Server 159 corrected-count and historical Top3 audit

Completed: 7 tasks, 3 H5 datasets, 42 filter combinations; 12 combinations below 0.8. All 21 historical Top3 candidates retain at least 0.8. All 7 tasks are NO_STEP3_NEEDED. No candidates are excluded from historical Top3; no replacements enter.

Step1 launches: 0. Step2 launches: 0. Step3 launches: 0. Pilot and trial monitoring are not applicable because no Top3 changed. No new joint best exists and no final result values may be updated from this audit. Historical Step3 results are not newly certified for runtime retention or trial completeness.

## Data and corrected counts

The requested analyzer ran with downloads enabled. Downloads: none; all three H5 files already existed in the highest-priority shared temp_data directory. find_h5 priority and download destination were verified. No MISSING_DATA; every task has six combinations. All X shapes match raw_cells/raw_genes; no dataset dimension conflicts. Ratios are computed as kept_cells/raw_cells; exactly 0.8 is retained.

| Dataset          | X shape         | H5 path                                                          | SHA256                                                           |
| ---------------- | --------------- | ---------------------------------------------------------------- | ---------------------------------------------------------------- |
| 10X_PBMC         | \[4271, 16653\] | /home/zyxing/dance/examples/tuning/temp_data/10X_PBMC.h5         | 34382b572c7d88aa19a7e0faf3a4be09393d3c2de2ccc617d781edea5f97d106 |
| human_ILCS_cell  | \[648, 64535\]  | /home/zyxing/dance/examples/tuning/temp_data/human_ILCS_cell.h5  | fa2d49b6120ed83d4baefa8bb8f71ab50c8f8cb1090bb73066759228a5e1e438 |
| worm_neuron_cell | \[4186, 13488\] | /home/zyxing/dance/examples/tuning/temp_data/worm_neuron_cell.h5 | 4f2a3c4b8492bcd11d957a8a74098ba258c271d714718c175854a7c44b1e31e6 |

Original CSV SHA256: `c6d572e0d1dd57802d9ac3cd090efdf03e223ff7b3891c60e3ab327694da9a1d`
Corrected CSV SHA256: `07a3c7a967df7ab973f22390efe8165a869f17baee87dbb8d39924f9a9a9f105`

All original/current count fields, H5 paths, filter-function keys and filter-operation details are identical after matching by method/dataset/gene_filter/cell_filter. The corrected CSV additionally stores the explicit retention ratio. The original CSV is preserved with its original bytes. No prior out_step3_cell_guard\*.log exists for these seven tasks, and no previous guarded run was reused.

## Combinations below 0.8

| Method        | Dataset         | Gene filter            | Cell filter            | Kept/raw  | Retention   |
| ------------- | --------------- | ---------------------- | ---------------------- | --------- | ----------- |
| SCDCC         | 10X_PBMC        | FilterGenesPercentile  | FilterCellsScanpyOrder | 1783/4271 | 0.417466635 |
| SCDCC         | 10X_PBMC        | FilterGenesScanpyOrder | FilterCellsScanpyOrder | 1746/4271 | 0.408803559 |
| SCDCC         | human_ILCS_cell | FilterGenesPercentile  | FilterCellsScanpyOrder | 480/648   | 0.740740741 |
| SCDCC         | human_ILCS_cell | FilterGenesScanpyOrder | FilterCellsScanpyOrder | 482/648   | 0.743827160 |
| SCDEEPCLUSTER | human_ILCS_cell | FilterGenesPercentile  | FilterCellsScanpyOrder | 480/648   | 0.740740741 |
| SCDEEPCLUSTER | human_ILCS_cell | FilterGenesScanpyOrder | FilterCellsScanpyOrder | 482/648   | 0.743827160 |
| SCDSC         | 10X_PBMC        | FilterGenesPercentile  | FilterCellsScanpyOrder | 1783/4271 | 0.417466635 |
| SCDSC         | 10X_PBMC        | FilterGenesScanpyOrder | FilterCellsScanpyOrder | 1746/4271 | 0.408803559 |
| SCDSC         | human_ILCS_cell | FilterGenesPercentile  | FilterCellsScanpyOrder | 480/648   | 0.740740741 |
| SCDSC         | human_ILCS_cell | FilterGenesScanpyOrder | FilterCellsScanpyOrder | 482/648   | 0.743827160 |
| SCTAG         | human_ILCS_cell | FilterGenesPercentile  | FilterCellsScanpyOrder | 480/648   | 0.740740741 |
| SCTAG         | human_ILCS_cell | FilterGenesScanpyOrder | FilterCellsScanpyOrder | 482/648   | 0.743827160 |

## Historical sweep provenance

Primary Step2 sweep IDs come exclusively from results - cluster.csv. Associated sweep IDs were resolved from historical wandb-metadata.json arguments using get_additional_sweep(). W&B run configs and summaries were downloaded without agents. All 4104 IDs, acc values (numeric tolerance 1e-12), and pipeline fields match local summaries; local originals were not rewritten. Raw W&B exports and full run configs/summaries are in step2_provenance/.

| Method        | Dataset          | Sweeps (primary first) | Verified runs |
| ------------- | ---------------- | ---------------------- | ------------- |
| SCDCC         | 10X_PBMC         | bl236xc7, 4qdza524     | 216           |
| SCDCC         | human_ILCS_cell  | gx9i91pb, cuqx61va     | 216           |
| SCDEEPCLUSTER | human_ILCS_cell  | m6yn75go, 0d0h4cnj     | 216           |
| SCDSC         | worm_neuron_cell | vqmhygcx, 2z7apgyl     | 864           |
| SCDSC         | 10X_PBMC         | n4j6jo73, 0wsjafv8     | 864           |
| SCDSC         | human_ILCS_cell  | 9o2yvv6q, g56cb2ac     | 864           |
| SCTAG         | human_ILCS_cell  | 81fqd0nb, msqj6fao     | 864           |

## Historical Top3 → guarded Top3

Historical ranking uses the repository’s original acc descending sort on the full CSV, once. Guarded ranking removes below-threshold rows from that same order. This preserves original tie order. Sorting again after filtering can spuriously swap tied candidates even when no historical Top3 candidate is excluded; dance/pipeline.py was corrected to rank before filtering. Two regression cases passed, including exactly 0.8 and replacement of an excluded winner. The preliminary audit directory from before this correction is retained but superseded by this audit.

| Method        | Dataset          | Rank | Historical ID → guarded ID | acc            | Retention   | Decision |
| ------------- | ---------------- | ---- | -------------------------- | -------------- | ----------- | -------- |
| SCDCC         | 10X_PBMC         | 1    | vt5nay0m → vt5nay0m        | 0.844467179538 | 0.984312807 | KEEP     |
| SCDCC         | 10X_PBMC         | 2    | 70qplttf → 70qplttf        | 0.837021077306 | 1.000000000 | KEEP     |
| SCDCC         | 10X_PBMC         | 3    | rgqn3vvo → rgqn3vvo        | 0.826216923571 | 1.000000000 | KEEP     |
| SCDCC         | human_ILCS_cell  | 1    | 8t78un2u → 8t78un2u        | 0.172261458621 | 1.000000000 | KEEP     |
| SCDCC         | human_ILCS_cell  | 2    | jb62ui8h → jb62ui8h        | 0.17106015438  | 1.000000000 | KEEP     |
| SCDCC         | human_ILCS_cell  | 3    | eemulfzs → eemulfzs        | 0.17106015438  | 1.000000000 | KEEP     |
| SCDEEPCLUSTER | human_ILCS_cell  | 1    | 8jie17h3 → 8jie17h3        | 0.541898384693 | 1.000000000 | KEEP     |
| SCDEEPCLUSTER | human_ILCS_cell  | 2    | wl59y7b6 → wl59y7b6        | 0.541898384693 | 1.000000000 | KEEP     |
| SCDEEPCLUSTER | human_ILCS_cell  | 3    | hbne3ukv → hbne3ukv        | 0.44259801067  | 1.000000000 | KEEP     |
| SCDSC         | worm_neuron_cell | 1    | to55sqp6 → to55sqp6        | 0.655493335464 | 0.983038700 | KEEP     |
| SCDSC         | worm_neuron_cell | 2    | h58u5oms → h58u5oms        | 0.655493335464 | 0.983038700 | KEEP     |
| SCDSC         | worm_neuron_cell | 3    | uetzgf3a → uetzgf3a        | 0.655493335464 | 0.983038700 | KEEP     |
| SCDSC         | 10X_PBMC         | 1    | 15m7hot1 → 15m7hot1        | 0.720369111377 | 1.000000000 | KEEP     |
| SCDSC         | 10X_PBMC         | 2    | rg8irg7b → rg8irg7b        | 0.720369111377 | 1.000000000 | KEEP     |
| SCDSC         | 10X_PBMC         | 3    | 3sbvxzv4 → 3sbvxzv4        | 0.720369111377 | 1.000000000 | KEEP     |
| SCDSC         | human_ILCS_cell  | 1    | mnccuzxb → mnccuzxb        | 0.6763138849   | 0.986111111 | KEEP     |
| SCDSC         | human_ILCS_cell  | 2    | i4t5qxul → i4t5qxul        | 0.6763138849   | 0.986111111 | KEEP     |
| SCDSC         | human_ILCS_cell  | 3    | vz7l1o7v → vz7l1o7v        | 0.6763138849   | 0.986111111 | KEEP     |
| SCTAG         | human_ILCS_cell  | 1    | hszjezax → hszjezax        | 0.668502600814 | 1.000000000 | KEEP     |
| SCTAG         | human_ILCS_cell  | 2    | 0nd39u6v → 0nd39u6v        | 0.668425492986 | 1.000000000 | KEEP     |
| SCTAG         | human_ILCS_cell  | 3    | 1ofbvzv6 → 1ofbvzv6        | 0.425030417605 | 0.986111111 | KEEP     |

## Execution, completeness and preservation

No NEEDS_STEP3 or BLOCKED tasks remain. No main.py was invoked, no new sweep was created, and no GPU was allocated. Consequently there is no pilot execution, no new incomplete task, and all newly planned/started/completed/valid/failed/guard-skipped trial counts are zero. CUDA OOM, NaN/Inf, Singular matrix, Killed and Traceback counts for new trials are not applicable. The initial sandboxed W&B probe encountered ProxyError; authorized network access succeeded for every sweep.

The mandatory “Skip step-2 sweep agent because count=0” marker applies to cluster executions; none were necessary. It is not fabricated as an observed training-log line. execution_audit.log records zero main.py/Step2 invocations, and the API-only verification script contains no agent launch.

old_joint_best contains only the observed maximum of existing local Step3 CSVs; historical_step3_observations.json records file hashes, row counts and numeric acc counts. These maxima are not completeness or runtime-guard certifications. new_joint_best is empty for every task. No result table is updated.

Archive: /home/zyxing/dance/examples/tuning/cluster_159_step3_archive/corrected_counts_20260906_020637
Initial original-artifact backup: /home/zyxing/dance/examples/tuning/cluster_159_step3_archive/corrected_counts_20260906_020609
No task results/config directories required execution archives because there are no NEEDS_STEP3 tasks. All original task results/configs/logs remain in place. New annotated Step2 files are separate from best_test_acc.csv. The archive manifest covers audit/provenance artifacts; each copied artifact is hash-compared to its source. No reset, checkout, deletion, commit or push was performed.
