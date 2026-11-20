# Cluster cell-filtering audit: server 205

## Executive conclusion

The source CSV contains 78 simulated preprocessing rows: six gene-filter/cell-filter combinations for each of 13 method/dataset assignments. Eight combinations have `cell_retention < 0.8`, covering four assignments: `Graphsc/10X_PBMC`, `Graphsc/human_ILCS_cell`, `SCDCC/mouse_kidney_cell`, and `SCTAG/10X_PBMC`.

There is no local command or log evidence that the historical step3 runs used `min_cell_retention=0.8`. All five current `main.py` files define the argument, but its default is `None`. The checked `run.sh` files do not pass it, and the existing logs contain neither the argument nor a filtering/guard event. Git blame dates the cluster entry-point support to 2026-09-03, whereas the audited logs and results are from 2024-2025. Therefore all five methods are classified as **parameter exists, but no evidence that historical step3 passed it**. No method meets the stronger “command plus log proof” category.

Applying the combination-level 0.8 rule to the stored step2 summaries changes the selected top three only for `SCDCC/mouse_kidney_cell`: old run `b2u000bw` (`FilterGenesScanpyOrder+FilterCellsScanpyOrder`, retention 0.5) is removed and `jbqiafa6` becomes the third eligible candidate. Its step3 selection/configuration and derived result need to be regenerated; at minimum `config_yamls/params/1_params_tuning_config.yaml` and `results/params/1_best.csv` are based on the excluded family. Regenerating all three indexed step3 configs/results for this one assignment is safer because filtering changes index ordering. For the other 12 assignments, the historical selected top three are already at or above 0.8 under this audit, so the threshold alone does not require model recomputation.

## Scope and calculation

`cell_retention` is calculated as `kept_cells / raw_cells` from `cluster_cell_filtering_205.csv`. The source script evaluates the default parameters in each `pipeline_params_tuning_config.yaml`; it does not replay every parameter value sampled during historical step3 sweeps. Consequently, the result is a combination-level eligibility audit, not a per-W&B-run reconstruction.

All seven CSV H5 paths currently exist, their `X` shapes match `raw_cells` and `raw_genes`, all 13 YAML paths exist, and no dataset has conflicting dimensions within the CSV. Existing historical logs also report the same raw dimensions for every assignment.

## Data-source validation

All current cluster `main.py` files default to `--data_dir ../temp_data`. When launched from the method directory as in `run.sh`, that resolves to `/home/common/zyxing/dance/examples/tuning/temp_data`, matching the CSV. Historical logs are stronger evidence for old runs and show these exceptions:

| method/dataset                  | CSV H5 and shape                                               | historical log data directory                                    | assessment                                                                                                                                                           |
| ------------------------------- | -------------------------------------------------------------- | ---------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Graphsc/10X_PBMC                | `examples/tuning/temp_data/10X_PBMC.h5`, 4271 x 16653          | method-local `./temp_data`; historical root `/home/zyxing/dance` | **CAUTION:** historical file is unavailable now. Log shape matches, but exact H5 identity/content cannot be verified, so historical retention is not fully reliable. |
| Graphsc/human_ILCS_cell         | `examples/tuning/temp_data/human_ILCS_cell.h5`, 648 x 64535    | method-local `./temp_data`; historical root `/home/zyxing/dance` | **CAUTION:** same limitation; log shape matches only.                                                                                                                |
| Graphsc/worm_neuron_cell        | `examples/tuning/temp_data/worm_neuron_cell.h5`, 4186 x 13488  | method-local `./temp_data`; historical root `/home/zyxing/dance` | **CAUTION:** same limitation; log shape matches only.                                                                                                                |
| SCDCC/human_skin_cell           | `examples/tuning/temp_data/human_skin_cell.h5`, 4853 x 1000    | `../temp_data`                                                   | Reliable path and dimension match.                                                                                                                                   |
| SCDCC/mouse_kidney_cell         | `examples/tuning/temp_data/mouse_kidney_cell.h5`, 3660 x 23797 | `../temp_data`                                                   | Reliable path and dimension match.                                                                                                                                   |
| SCDCC/mouse_kidney_cl2          | `examples/tuning/temp_data/mouse_kidney_cl2.h5`, 274 x 28204   | `../temp_data`                                                   | Reliable path and dimension match.                                                                                                                                   |
| SCDCC/mouse_kidney_drop         | `examples/tuning/temp_data/mouse_kidney_drop.h5`, 225 x 15127  | `../temp_data`                                                   | Reliable path and dimension match.                                                                                                                                   |
| SCDEEPCLUSTER/mouse_kidney_drop | `examples/tuning/temp_data/mouse_kidney_drop.h5`, 225 x 15127  | `../temp_data`                                                   | Reliable path and dimension match.                                                                                                                                   |
| SCDSC/mouse_kidney_drop         | `examples/tuning/temp_data/mouse_kidney_drop.h5`, 225 x 15127  | method-local `./data`                                            | Reliable for this audit: the current method-local and shared H5 files have identical SHA-256 `114d1f91f7c673bc6ff4de9cc4285478cc7e207f4b2c8705a517f58e132622d9`.     |
| SCTAG/10X_PBMC                  | `examples/tuning/temp_data/10X_PBMC.h5`, 4271 x 16653          | `../temp_data`                                                   | Reliable path and dimension match.                                                                                                                                   |
| SCTAG/mouse_kidney_cl2          | `examples/tuning/temp_data/mouse_kidney_cl2.h5`, 274 x 28204   | `../temp_data`                                                   | Reliable path and dimension match.                                                                                                                                   |
| SCTAG/mouse_kidney_drop         | `examples/tuning/temp_data/mouse_kidney_drop.h5`, 225 x 15127  | method-local `./data`                                            | Reliable for this audit: its H5 has the same SHA-256 as the shared H5.                                                                                               |
| SCTAG/worm_neuron_cell          | `examples/tuning/temp_data/worm_neuron_cell.h5`, 4186 x 13488  | `../temp_data`                                                   | Reliable path and dimension match.                                                                                                                                   |

There are unused method-local `mouse_kidney_cl2.h5` files with shape 225 x 15127, conflicting with the shared file's 274 x 28204. They are not treated as inputs for the server-205 assignments because the corresponding SCDCC and SCTAG logs explicitly use `../temp_data`. This collision reinforces the need to preserve the documented H5 search priority.

## Combinations below 0.8

These eight combinations would be excluded if the 0.8 threshold were enabled:

| method  | dataset           | scenario                                      | gene_filter            | cell_filter            | raw_cells | kept_cells | removed_cells | cell_retention | H5                                                                         | YAML                                                                                                           |
| ------- | ----------------- | --------------------------------------------- | ---------------------- | ---------------------- | --------: | ---------: | ------------: | -------------: | -------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| Graphsc | 10X_PBMC          | FilterGenesScanpyOrder+FilterCellsScanpyOrder | FilterGenesScanpyOrder | FilterCellsScanpyOrder |      4271 |       1746 |          2525 |       0.408804 | `/home/common/zyxing/dance/examples/tuning/temp_data/10X_PBMC.h5`          | `/home/common/zyxing/dance/examples/tuning/cluster_graphsc/10X_PBMC/pipeline_params_tuning_config.yaml`        |
| Graphsc | 10X_PBMC          | FilterGenesPercentile+FilterCellsScanpyOrder  | FilterGenesPercentile  | FilterCellsScanpyOrder |      4271 |       1783 |          2488 |       0.417467 | `/home/common/zyxing/dance/examples/tuning/temp_data/10X_PBMC.h5`          | `/home/common/zyxing/dance/examples/tuning/cluster_graphsc/10X_PBMC/pipeline_params_tuning_config.yaml`        |
| Graphsc | human_ILCS_cell   | FilterGenesPercentile+FilterCellsScanpyOrder  | FilterGenesPercentile  | FilterCellsScanpyOrder |       648 |        480 |           168 |       0.740741 | `/home/common/zyxing/dance/examples/tuning/temp_data/human_ILCS_cell.h5`   | `/home/common/zyxing/dance/examples/tuning/cluster_graphsc/human_ILCS_cell/pipeline_params_tuning_config.yaml` |
| Graphsc | human_ILCS_cell   | FilterGenesScanpyOrder+FilterCellsScanpyOrder | FilterGenesScanpyOrder | FilterCellsScanpyOrder |       648 |        482 |           166 |       0.743827 | `/home/common/zyxing/dance/examples/tuning/temp_data/human_ILCS_cell.h5`   | `/home/common/zyxing/dance/examples/tuning/cluster_graphsc/human_ILCS_cell/pipeline_params_tuning_config.yaml` |
| SCDCC   | mouse_kidney_cell | FilterGenesScanpyOrder+FilterCellsScanpyOrder | FilterGenesScanpyOrder | FilterCellsScanpyOrder |      3660 |       1830 |          1830 |       0.500000 | `/home/common/zyxing/dance/examples/tuning/temp_data/mouse_kidney_cell.h5` | `/home/common/zyxing/dance/examples/tuning/cluster_scdcc/mouse_kidney_cell/pipeline_params_tuning_config.yaml` |
| SCDCC   | mouse_kidney_cell | FilterGenesPercentile+FilterCellsScanpyOrder  | FilterGenesPercentile  | FilterCellsScanpyOrder |      3660 |       1857 |          1803 |       0.507377 | `/home/common/zyxing/dance/examples/tuning/temp_data/mouse_kidney_cell.h5` | `/home/common/zyxing/dance/examples/tuning/cluster_scdcc/mouse_kidney_cell/pipeline_params_tuning_config.yaml` |
| SCTAG   | 10X_PBMC          | FilterGenesScanpyOrder+FilterCellsScanpyOrder | FilterGenesScanpyOrder | FilterCellsScanpyOrder |      4271 |       1746 |          2525 |       0.408804 | `/home/common/zyxing/dance/examples/tuning/temp_data/10X_PBMC.h5`          | `/home/common/zyxing/dance/examples/tuning/cluster_sctag/10X_PBMC/pipeline_params_tuning_config.yaml`          |
| SCTAG   | 10X_PBMC          | FilterGenesPercentile+FilterCellsScanpyOrder  | FilterGenesPercentile  | FilterCellsScanpyOrder |      4271 |       1783 |          2488 |       0.417467 | `/home/common/zyxing/dance/examples/tuning/temp_data/10X_PBMC.h5`          | `/home/common/zyxing/dance/examples/tuning/cluster_sctag/10X_PBMC/pipeline_params_tuning_config.yaml`          |

The three Graphsc rows in this section are subject to the historical-H5 caution above. They are exact for the currently available shared H5, but should not be asserted as exact historical-run retention without recovering or hashing the old method-local files.

## Per-assignment summary

| method/dataset                  | combinations | below 0.8 | minimum retention | worst combination                             | raw cells | minimum kept |
| ------------------------------- | -----------: | --------: | ----------------: | --------------------------------------------- | --------: | -----------: |
| Graphsc/10X_PBMC                |            6 |         2 |          0.408804 | FilterGenesScanpyOrder+FilterCellsScanpyOrder |      4271 |         1746 |
| Graphsc/human_ILCS_cell         |            6 |         2 |          0.740741 | FilterGenesPercentile+FilterCellsScanpyOrder  |       648 |          480 |
| Graphsc/worm_neuron_cell        |            6 |         0 |          0.834209 | FilterGenesScanpyOrder+FilterCellsScanpyOrder |      4186 |         3492 |
| SCDCC/human_skin_cell           |            6 |         0 |          0.913662 | FilterGenesScanpyOrder+FilterCellsScanpyOrder |      4853 |         4434 |
| SCDCC/mouse_kidney_cell         |            6 |         2 |          0.500000 | FilterGenesScanpyOrder+FilterCellsScanpyOrder |      3660 |         1830 |
| SCDCC/mouse_kidney_cl2          |            6 |         0 |          0.890511 | FilterGenesScanpyOrder+FilterCellsScanpyOrder |       274 |          244 |
| SCDCC/mouse_kidney_drop         |            6 |         0 |          0.826667 | FilterGenesScanpyOrder+FilterCellsScanpyOrder |       225 |          186 |
| SCDEEPCLUSTER/mouse_kidney_drop |            6 |         0 |          0.826667 | FilterGenesScanpyOrder+FilterCellsScanpyOrder |       225 |          186 |
| SCDSC/mouse_kidney_drop         |            6 |         0 |          0.826667 | FilterGenesScanpyOrder+FilterCellsScanpyOrder |       225 |          186 |
| SCTAG/10X_PBMC                  |            6 |         2 |          0.408804 | FilterGenesScanpyOrder+FilterCellsScanpyOrder |      4271 |         1746 |
| SCTAG/mouse_kidney_cl2          |            6 |         0 |          0.890511 | FilterGenesScanpyOrder+FilterCellsScanpyOrder |       274 |          244 |
| SCTAG/mouse_kidney_drop         |            6 |         0 |          0.826667 | FilterGenesScanpyOrder+FilterCellsScanpyOrder |       225 |          186 |
| SCTAG/worm_neuron_cell          |            6 |         0 |          0.834209 | FilterGenesScanpyOrder+FilterCellsScanpyOrder |      4186 |         3492 |

## Historical `min_cell_retention=0.8` evidence

| method        | classification                                      | evidence                                                                                                                                                                                 |
| ------------- | --------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Graphsc       | Parameter exists; no historical enablement evidence | Current parser default is `None`; `run.sh` does not pass the flag; logs contain no argument, “Filtered step-2 pipelines”, skip, or guard event.                                          |
| SCDCC         | Parameter exists; no historical enablement evidence | Same. The low-retention `mouse_kidney_cell` family nevertheless entered the historical top three, which is direct outcome evidence that combination-level 0.8 filtering was not applied. |
| SCDEEPCLUSTER | Parameter exists; no historical enablement evidence | Current support exists, but commands/logs provide no proof of use.                                                                                                                       |
| SCDSC         | Parameter exists; no historical enablement evidence | Current support exists, but commands/logs provide no proof of use.                                                                                                                       |
| SCTAG         | Parameter exists; no historical enablement evidence | Current support exists, but commands/logs provide no proof of use.                                                                                                                       |

The helper `generate_cluster_112_step3_commands.py` can generate commands with a default 0.8 threshold, but it targets server 112, not server 205, and a generator's existence is not execution evidence. No generated `run_cluster_*step3*` script or `out_step3_cell_guard.log` was found locally.

## Recalculation and limitations

- **Recalculate/regenerate:** `SCDCC/mouse_kidney_cell` step3 selection and the result associated with the excluded old second-ranked candidate (`b2u000bw`; existing index 1). Rebuild the three step3 indices if a coherent guarded top-three result set is required. No training was started by this audit.
- **No threshold-only recalculation indicated:** the other 12 assignments, because their stored top-three step2 candidates all have audited combination retention at least 0.8.
- **Needs source verification, not an unconditional model rerun:** the three Graphsc assignments. Recover the historical method-local H5 files or establish content hashes before treating shared-H5 retention as an exact reconstruction. Their logged dimensions agree, and no low-retention Graphsc combination entered the historical top three.
- **Log coverage limitation:** this conclusion covers commands and logs present in this checkout. It does not prove that no unarchived remote invocation ever supplied the threshold. The available artifacts do prove that the checked historical outputs predate the current entry-point argument and contain no guard evidence.
- **Parameter-level limitation:** retention values are based on pipeline YAML defaults. Historical step3 parameter trials may have varied filter thresholds; per-run recomputation would require replaying each run's preprocessing parameters or retaining logged per-run cell counts.
