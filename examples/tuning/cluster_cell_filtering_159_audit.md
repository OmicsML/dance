# Cluster cell-filtering audit: server 159

## Scope and conclusion

This audit covers the 42 preprocessing rows in `examples/tuning/cluster_cell_filtering_159.csv`: 4 methods, 3 datasets, and 7 method/dataset pairs. There are 12 combinations with `cell_retention < 0.8`.

The historical step3 runs cannot be claimed to have used `min_cell_retention=0.8`. All five current `cluster_*` entry points define the parameter, but its default is `None`; no checked `cluster_*` launch command passes it, and no checked log contains a filtering or cell-count-guard marker. The guard-supporting code was added in commit `78ba0f4` on 2026-09-03, after the inspected 2024-2025 runs.

## Data-source validation

All H5 and YAML paths referenced by the input CSV exist. Each H5 has an `X` dataset whose shape agrees with `raw_cells` and `raw_genes` in every corresponding CSV row.

| Dataset            | CSV H5                                          | raw_cells | raw_genes | Validation |
| ------------------ | ----------------------------------------------- | --------: | --------: | ---------- |
| `10X_PBMC`         | `examples/tuning/temp_data/10X_PBMC.h5`         |      4271 |     16653 | Reliable   |
| `human_ILCS_cell`  | `examples/tuning/temp_data/human_ILCS_cell.h5`  |       648 |     64535 | Reliable   |
| `worm_neuron_cell` | `examples/tuning/temp_data/worm_neuron_cell.h5` |      4186 |     13488 | Reliable   |

The current `main.py` default is `--data_dir ../temp_data` for Graphsc, SCDCC, SCDEEPCLUSTER, SCDSC, and SCTAG. With the launch convention shown by each method's `run.sh` (run from the method directory), this resolves to `examples/tuning/temp_data`, matching the CSV.

The inspected 2025 logs for the seven server-159 pairs use `../temp_data`, except that older SCDEEPCLUSTER/human_ILCS_cell and SCDSC/human_ILCS_cell log sections from 2024 used `./data`. The local and shared copies are byte-identical:

- `human_ILCS_cell.h5`: SHA-256 `fa2d49b6120ed83d4baefa8bb8f71ab50c8f8cb1090bb73066759228a5e1e438`
- `10X_PBMC.h5`: SHA-256 `34382b572c7d88aa19a7e0faf3a4be09393d3c2de2ccc617d781edea5f97d106`
- `worm_neuron_cell.h5`: SHA-256 `4f2a3c4b8492bcd11d957a8a74098ba258c271d714718c175854a7c44b1e31e6`

No dataset has conflicting original dimensions, and no data-source row is marked suspicious. The historical location change is documented but does not alter this audit because the compared files are identical.

## Combinations below 0.8

`cell_retention` is calculated as `kept_cells / raw_cells`. If a 0.8 guard is enabled, all 12 rows below must be excluded.

| method        | dataset         | scenario                                      | gene_filter            | cell_filter            | raw_cells | kept_cells | removed_cells | cell_retention | h5                                             | yaml                                                                                       |
| ------------- | --------------- | --------------------------------------------- | ---------------------- | ---------------------- | --------: | ---------: | ------------: | -------------: | ---------------------------------------------- | ------------------------------------------------------------------------------------------ |
| SCDCC         | 10X_PBMC        | FilterGenesScanpyOrder+FilterCellsScanpyOrder | FilterGenesScanpyOrder | FilterCellsScanpyOrder |      4271 |       1746 |          2525 |       0.408804 | `examples/tuning/temp_data/10X_PBMC.h5`        | `examples/tuning/cluster_scdcc/10X_PBMC/pipeline_params_tuning_config.yaml`                |
| SCDCC         | 10X_PBMC        | FilterGenesPercentile+FilterCellsScanpyOrder  | FilterGenesPercentile  | FilterCellsScanpyOrder |      4271 |       1783 |          2488 |       0.417467 | `examples/tuning/temp_data/10X_PBMC.h5`        | `examples/tuning/cluster_scdcc/10X_PBMC/pipeline_params_tuning_config.yaml`                |
| SCDCC         | human_ILCS_cell | FilterGenesPercentile+FilterCellsScanpyOrder  | FilterGenesPercentile  | FilterCellsScanpyOrder |       648 |        480 |           168 |       0.740741 | `examples/tuning/temp_data/human_ILCS_cell.h5` | `examples/tuning/cluster_scdcc/human_ILCS_cell/pipeline_params_tuning_config.yaml`         |
| SCDCC         | human_ILCS_cell | FilterGenesScanpyOrder+FilterCellsScanpyOrder | FilterGenesScanpyOrder | FilterCellsScanpyOrder |       648 |        482 |           166 |       0.743827 | `examples/tuning/temp_data/human_ILCS_cell.h5` | `examples/tuning/cluster_scdcc/human_ILCS_cell/pipeline_params_tuning_config.yaml`         |
| SCDEEPCLUSTER | human_ILCS_cell | FilterGenesPercentile+FilterCellsScanpyOrder  | FilterGenesPercentile  | FilterCellsScanpyOrder |       648 |        480 |           168 |       0.740741 | `examples/tuning/temp_data/human_ILCS_cell.h5` | `examples/tuning/cluster_scdeepcluster/human_ILCS_cell/pipeline_params_tuning_config.yaml` |
| SCDEEPCLUSTER | human_ILCS_cell | FilterGenesScanpyOrder+FilterCellsScanpyOrder | FilterGenesScanpyOrder | FilterCellsScanpyOrder |       648 |        482 |           166 |       0.743827 | `examples/tuning/temp_data/human_ILCS_cell.h5` | `examples/tuning/cluster_scdeepcluster/human_ILCS_cell/pipeline_params_tuning_config.yaml` |
| SCDSC         | 10X_PBMC        | FilterGenesScanpyOrder+FilterCellsScanpyOrder | FilterGenesScanpyOrder | FilterCellsScanpyOrder |      4271 |       1746 |          2525 |       0.408804 | `examples/tuning/temp_data/10X_PBMC.h5`        | `examples/tuning/cluster_scdsc/10X_PBMC/pipeline_params_tuning_config.yaml`                |
| SCDSC         | 10X_PBMC        | FilterGenesPercentile+FilterCellsScanpyOrder  | FilterGenesPercentile  | FilterCellsScanpyOrder |      4271 |       1783 |          2488 |       0.417467 | `examples/tuning/temp_data/10X_PBMC.h5`        | `examples/tuning/cluster_scdsc/10X_PBMC/pipeline_params_tuning_config.yaml`                |
| SCDSC         | human_ILCS_cell | FilterGenesPercentile+FilterCellsScanpyOrder  | FilterGenesPercentile  | FilterCellsScanpyOrder |       648 |        480 |           168 |       0.740741 | `examples/tuning/temp_data/human_ILCS_cell.h5` | `examples/tuning/cluster_scdsc/human_ILCS_cell/pipeline_params_tuning_config.yaml`         |
| SCDSC         | human_ILCS_cell | FilterGenesScanpyOrder+FilterCellsScanpyOrder | FilterGenesScanpyOrder | FilterCellsScanpyOrder |       648 |        482 |           166 |       0.743827 | `examples/tuning/temp_data/human_ILCS_cell.h5` | `examples/tuning/cluster_scdsc/human_ILCS_cell/pipeline_params_tuning_config.yaml`         |
| SCTAG         | human_ILCS_cell | FilterGenesPercentile+FilterCellsScanpyOrder  | FilterGenesPercentile  | FilterCellsScanpyOrder |       648 |        480 |           168 |       0.740741 | `examples/tuning/temp_data/human_ILCS_cell.h5` | `examples/tuning/cluster_sctag/human_ILCS_cell/pipeline_params_tuning_config.yaml`         |
| SCTAG         | human_ILCS_cell | FilterGenesScanpyOrder+FilterCellsScanpyOrder | FilterGenesScanpyOrder | FilterCellsScanpyOrder |       648 |        482 |           166 |       0.743827 | `examples/tuning/temp_data/human_ILCS_cell.h5` | `examples/tuning/cluster_sctag/human_ILCS_cell/pipeline_params_tuning_config.yaml`         |

Thus, low-retention candidates occur for SCDCC/10X_PBMC, SCDCC/human_ILCS_cell, SCDEEPCLUSTER/human_ILCS_cell, SCDSC/10X_PBMC, SCDSC/human_ILCS_cell, and SCTAG/human_ILCS_cell. SCDSC/worm_neuron_cell has none; its minimum is 0.834209.

## Method/dataset summary

The machine-readable aggregate is in `examples/tuning/cluster_cell_filtering_159_summary.csv`.

| method/dataset                | combinations | below 0.8 | minimum retention | minimum kept cells | worst combination                             |
| ----------------------------- | -----------: | --------: | ----------------: | -----------------: | --------------------------------------------- |
| SCDCC/10X_PBMC                |            6 |         2 |          0.408804 |               1746 | FilterGenesScanpyOrder+FilterCellsScanpyOrder |
| SCDCC/human_ILCS_cell         |            6 |         2 |          0.740741 |                480 | FilterGenesPercentile+FilterCellsScanpyOrder  |
| SCDEEPCLUSTER/human_ILCS_cell |            6 |         2 |          0.740741 |                480 | FilterGenesPercentile+FilterCellsScanpyOrder  |
| SCDSC/10X_PBMC                |            6 |         2 |          0.408804 |               1746 | FilterGenesScanpyOrder+FilterCellsScanpyOrder |
| SCDSC/human_ILCS_cell         |            6 |         2 |          0.740741 |                480 | FilterGenesPercentile+FilterCellsScanpyOrder  |
| SCDSC/worm_neuron_cell        |            6 |         0 |          0.834209 |               3492 | FilterGenesScanpyOrder+FilterCellsScanpyOrder |
| SCTAG/human_ILCS_cell         |            6 |         2 |          0.740741 |                480 | FilterGenesPercentile+FilterCellsScanpyOrder  |

## Historical step3 threshold evidence

Classification is based on checked launch commands and logs, not merely on current argparse definitions.

| Method        | Classification                                               | Evidence                                                                                                                          |
| ------------- | ------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------- |
| Graphsc       | Parameter exists, but no evidence it was passed historically | Current default is `None`; `run.sh` does not pass it; no guard/filter log marker. Graphsc has no server-159 row in the input CSV. |
| SCDCC         | Parameter exists, but no evidence it was passed historically | Current default is `None`; historical parameter dumps do not contain it; `run.sh` does not pass it; no guard/filter log marker.   |
| SCDEEPCLUSTER | Parameter exists, but no evidence it was passed historically | Same evidence; no historical command or log establishes 0.8.                                                                      |
| SCDSC         | Parameter exists, but no evidence it was passed historically | Same evidence; no historical command or log establishes 0.8.                                                                      |
| SCTAG         | Parameter exists, but no evidence it was passed historically | Same evidence; no historical command or log establishes 0.8.                                                                      |

No method qualifies for “explicitly passed 0.8 and log proves filtering.” No method lacks support completely in the current code. `generate_cluster_112_step3_commands.py` has a default of 0.8 and renders guarded commands, but it targets server 112; there is no generated server-159 guarded command or corresponding `out_step3_cell_guard.log`, so it is not evidence for these historical runs.

## Results requiring action

To make a defensible claim that historical server-159 step3 results obey `cell_retention >= 0.8`, step3 candidate selection must be regenerated for all seven method/dataset pairs with the guard explicitly enabled and logged. This does not automatically mean every model must be retrained: retraining is required where guarded selection differs from the historical selection.

The six method/dataset pairs containing low-retention candidates are the priority for re-selection and possible recomputation:

- SCDCC/10X_PBMC
- SCDCC/human_ILCS_cell
- SCDEEPCLUSTER/human_ILCS_cell
- SCDSC/10X_PBMC
- SCDSC/human_ILCS_cell
- SCTAG/human_ILCS_cell

SCDSC/worm_neuron_cell has no below-0.8 combination under the audited defaults, so the CSV alone does not indicate a filtering-driven need to retrain it. However, the historical run still lacks proof that the guard was enabled.

Historical step3 parameter sweeps tune filtering thresholds, whereas this input CSV evaluates the six pipeline combinations at the parent YAML defaults. Therefore the CSV is sufficient to exclude the 12 listed default combinations, but it is not sufficient to certify every individual historical step3 run. Existing parameter-level results must not be declared valid or invalid without reconstructing their realized retention. Missing guard logs are the limiting evidence; H5 data are not missing.
