# papermachine corrected-counts / Step3 审计

更新时间 UTC：2026-09-06T00:58:26.204130+00:00。运行状态由后台控制器更新。

## 结论

- 17 个服务器任务、5 个数据集、102 种过滤组合；10 种低于 0.8。4 个 Top3 发生变化，13 个 NO_STEP3_NEEDED。
- 本次 Step1 agent 数量为 0，Step2 agent 数量为 0。预检查只通过 W&B API读取已有 sweep；实际 main.py 命令均 count=0。
- 正式细胞 CSV 已校正；旧 CSV 的 17 行均缺失，不能复用为计数依据。未找到 4 个变更任务的既有 out_step3_cell_guard\*.log；不复用旧 guarded Step3。
- 只使用 API 验证过的历史候选。17 份本地 Step2 CSV 与关联 sweep 导出的 acc、pipeline、ID及行序完全一致，因此比较优先使用本地原表。所有原 Step2 表保持不变，main.py 新下载表写入 best_test_acc.corrected_step2.csv。
- 并列 acc 先保留原表的历史排序，再剔除低于 0.8 的候选；已修正原生成器“过滤后重新排序”改变并列名次的问题。
- 完整性采取保守规则：三个 sweep 各有完整计划数量且全部为 finished、有限非负 acc 才标 COMPLETE 并提供 new_joint_best。guard skip 单独统计；少量有效结果不能更新正式最优值。未启动任务的 valid_trials=0 表示本次没有 trial，不是历史无结果。

## 数据文件

本次自动下载下列 5 个 H5 到 `/egr/research-dselab/dingjia5/zhongyu/dance/examples/tuning/temp_data`；X.shape 与每条 raw_cells/raw_genes 一致，无同名维度冲突。
正式 CSV SHA256：`2cf534d9cf48c54c6fbae1e4e55f2d1c6f168ebe6136f88a7692c21107418468`。

| dataset            | shape (cells, genes) | SHA256                                                           | 绝对路径                                                                                    |
| ------------------ | -------------------- | ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| mouse_ES_cell      | \[2717, 24175\]      | f20cbbe9dc3e62eb2808d7bcce71b5b1793ea7a5f9c0e84b79dfb73c9249f916 | /egr/research-dselab/dingjia5/zhongyu/dance/examples/tuning/temp_data/mouse_ES_cell.h5      |
| mouse_bladder_cell | \[2746, 20670\]      | 4161c18ce79341788ee420157ddc4806af09c1e20209d18064a8ae1393630140 | /egr/research-dselab/dingjia5/zhongyu/dance/examples/tuning/temp_data/mouse_bladder_cell.h5 |
| mouse_kidney_10x   | \[902, 16468\]       | f6dc78a87f906f24f0440cbaa050945e6f2a8948db3fef9bde82340ffda52428 | /egr/research-dselab/dingjia5/zhongyu/dance/examples/tuning/temp_data/mouse_kidney_10x.h5   |
| mouse_lung_cell    | \[1756, 1000\]       | 6cea80e45cbffca96baea7f514cc259d71de4d6c3c7751ec57748151361f2ace | /egr/research-dselab/dingjia5/zhongyu/dance/examples/tuning/temp_data/mouse_lung_cell.h5    |
| worm_neuron_cell   | \[4186, 13488\]      | 4f2a3c4b8492bcd11d957a8a74098ba258c271d714718c175854a7c44b1e31e6 | /egr/research-dselab/dingjia5/zhongyu/dance/examples/tuning/temp_data/worm_neuron_cell.h5   |

旧 CSV/审计文件先复制归档、校验 SHA256 后才更新；新旧哈希见 provenance.json 和根 archive_manifest.sha256。

## 所有低保留率组合

| method        | dataset          | scenario                                      | gene_filter            | cell_filter            | raw_cells | kept_cells | removed_cells | cell_retention     | h5                                            | yaml                                                                                      |
| ------------- | ---------------- | --------------------------------------------- | ---------------------- | ---------------------- | --------- | ---------- | ------------- | ------------------ | --------------------------------------------- | ----------------------------------------------------------------------------------------- |
| Graphsc       | mouse_kidney_10x | FilterGenesPercentile+FilterCellsScanpyOrder  | FilterGenesPercentile  | FilterCellsScanpyOrder | 902       | 607        | 295           | 0.6729490022172949 | examples/tuning/temp_data/mouse_kidney_10x.h5 | examples/tuning/cluster_graphsc/mouse_kidney_10x/pipeline_params_tuning_config.yaml       |
| Graphsc       | mouse_kidney_10x | FilterGenesScanpyOrder+FilterCellsScanpyOrder | FilterGenesScanpyOrder | FilterCellsScanpyOrder | 902       | 0          | 902           | 0.0                | examples/tuning/temp_data/mouse_kidney_10x.h5 | examples/tuning/cluster_graphsc/mouse_kidney_10x/pipeline_params_tuning_config.yaml       |
| SCDCC         | mouse_kidney_10x | FilterGenesPercentile+FilterCellsScanpyOrder  | FilterGenesPercentile  | FilterCellsScanpyOrder | 902       | 607        | 295           | 0.6729490022172949 | examples/tuning/temp_data/mouse_kidney_10x.h5 | examples/tuning/cluster_scdcc/mouse_kidney_10x/pipeline_params_tuning_config.yaml         |
| SCDCC         | mouse_kidney_10x | FilterGenesScanpyOrder+FilterCellsScanpyOrder | FilterGenesScanpyOrder | FilterCellsScanpyOrder | 902       | 0          | 902           | 0.0                | examples/tuning/temp_data/mouse_kidney_10x.h5 | examples/tuning/cluster_scdcc/mouse_kidney_10x/pipeline_params_tuning_config.yaml         |
| SCDEEPCLUSTER | mouse_kidney_10x | FilterGenesPercentile+FilterCellsScanpyOrder  | FilterGenesPercentile  | FilterCellsScanpyOrder | 902       | 607        | 295           | 0.6729490022172949 | examples/tuning/temp_data/mouse_kidney_10x.h5 | examples/tuning/cluster_scdeepcluster/mouse_kidney_10x/pipeline_params_tuning_config.yaml |
| SCDEEPCLUSTER | mouse_kidney_10x | FilterGenesScanpyOrder+FilterCellsScanpyOrder | FilterGenesScanpyOrder | FilterCellsScanpyOrder | 902       | 0          | 902           | 0.0                | examples/tuning/temp_data/mouse_kidney_10x.h5 | examples/tuning/cluster_scdeepcluster/mouse_kidney_10x/pipeline_params_tuning_config.yaml |
| SCDSC         | mouse_kidney_10x | FilterGenesPercentile+FilterCellsScanpyOrder  | FilterGenesPercentile  | FilterCellsScanpyOrder | 902       | 607        | 295           | 0.6729490022172949 | examples/tuning/temp_data/mouse_kidney_10x.h5 | examples/tuning/cluster_scdsc/mouse_kidney_10x/pipeline_params_tuning_config.yaml         |
| SCDSC         | mouse_kidney_10x | FilterGenesScanpyOrder+FilterCellsScanpyOrder | FilterGenesScanpyOrder | FilterCellsScanpyOrder | 902       | 0          | 902           | 0.0                | examples/tuning/temp_data/mouse_kidney_10x.h5 | examples/tuning/cluster_scdsc/mouse_kidney_10x/pipeline_params_tuning_config.yaml         |
| SCTAG         | mouse_kidney_10x | FilterGenesPercentile+FilterCellsScanpyOrder  | FilterGenesPercentile  | FilterCellsScanpyOrder | 902       | 607        | 295           | 0.6729490022172949 | examples/tuning/temp_data/mouse_kidney_10x.h5 | examples/tuning/cluster_sctag/mouse_kidney_10x/pipeline_params_tuning_config.yaml         |
| SCTAG         | mouse_kidney_10x | FilterGenesScanpyOrder+FilterCellsScanpyOrder | FilterGenesScanpyOrder | FilterCellsScanpyOrder | 902       | 0          | 902           | 0.0                | examples/tuning/temp_data/mouse_kidney_10x.h5 | examples/tuning/cluster_sctag/mouse_kidney_10x/pipeline_params_tuning_config.yaml         |

严格采用 kept_cells/raw_cells \< 0.8 排除，恰好 0.8 保留。仅统计第一组 gene/cell 过滤的组合；Step3 参数改变时继续运行时 guard。

## Top3 对比

| method        | dataset            | 分类            | historical_top3 → guarded_top3                                                  | 排除           | 顺延           | Step2 N→M |
| ------------- | ------------------ | --------------- | ------------------------------------------------------------------------------- | -------------- | -------------- | --------- |
| Graphsc       | mouse_bladder_cell | NO_STEP3_NEEDED | \['qn5xuih0', 'vnopivig', '8k4zrd3y'\] → \['qn5xuih0', 'vnopivig', '8k4zrd3y'\] | \[\]           | \[\]           | 432→432   |
| Graphsc       | mouse_ES_cell      | NO_STEP3_NEEDED | \['0hnlk0k8', 'x4psgq67', '61f1z0ip'\] → \['0hnlk0k8', 'x4psgq67', '61f1z0ip'\] | \[\]           | \[\]           | 432→432   |
| Graphsc       | mouse_lung_cell    | NO_STEP3_NEEDED | \['88p5ht3m', '4veay8g8', 'm7ngq46a'\] → \['88p5ht3m', '4veay8g8', 'm7ngq46a'\] | \[\]           | \[\]           | 432→432   |
| Graphsc       | mouse_kidney_10x   | NEEDS_STEP3     | \['cko91i46', 'phmwaauw', '80rrf4kw'\] → \['phmwaauw', '80rrf4kw', 'u6uxpd14'\] | \['cko91i46'\] | \['u6uxpd14'\] | 432→288   |
| SCDCC         | worm_neuron_cell   | NO_STEP3_NEEDED | \['3w6tt0pz', 'wec4q957', '0ykod7uz'\] → \['3w6tt0pz', 'wec4q957', '0ykod7uz'\] | \[\]           | \[\]           | 216→216   |
| SCDCC         | mouse_bladder_cell | NO_STEP3_NEEDED | \['sc50aaj2', 'cpja8wcl', 'akg7tx4t'\] → \['sc50aaj2', 'cpja8wcl', 'akg7tx4t'\] | \[\]           | \[\]           | 216→216   |
| SCDCC         | mouse_ES_cell      | NO_STEP3_NEEDED | \['9p6zv581', '3oyfedmd', 'rv9hp6nz'\] → \['9p6zv581', '3oyfedmd', 'rv9hp6nz'\] | \[\]           | \[\]           | 216→216   |
| SCDCC         | mouse_lung_cell    | NO_STEP3_NEEDED | \['426wb3ym', 'rhu68lz5', 'funhg67u'\] → \['426wb3ym', 'rhu68lz5', 'funhg67u'\] | \[\]           | \[\]           | 216→216   |
| SCDCC         | mouse_kidney_10x   | NO_STEP3_NEEDED | \['9tml3fa3', '0nxsgyo0', 'c8nof7uf'\] → \['9tml3fa3', '0nxsgyo0', 'c8nof7uf'\] | \[\]           | \[\]           | 216→144   |
| SCDEEPCLUSTER | mouse_lung_cell    | NO_STEP3_NEEDED | \['50rl8jxg', 'y74hcpod', '3cmpib8x'\] → \['50rl8jxg', 'y74hcpod', '3cmpib8x'\] | \[\]           | \[\]           | 216→216   |
| SCDEEPCLUSTER | mouse_kidney_10x   | NEEDS_STEP3     | \['uxjzabtr', 'sb3qs08r', 'prz6k2w4'\] → \['uxjzabtr', 'sb3qs08r', 'myesxyl6'\] | \['prz6k2w4'\] | \['myesxyl6'\] | 216→144   |
| SCDSC         | mouse_bladder_cell | NO_STEP3_NEEDED | \['ygiakhrj', 'xypyogsm', 'iyu5b3wa'\] → \['ygiakhrj', 'xypyogsm', 'iyu5b3wa'\] | \[\]           | \[\]           | 864→864   |
| SCDSC         | mouse_ES_cell      | NO_STEP3_NEEDED | \['isx66l84', 'rkggm3l8', 'gnkr7acl'\] → \['isx66l84', 'rkggm3l8', 'gnkr7acl'\] | \[\]           | \[\]           | 864→864   |
| SCDSC         | mouse_lung_cell    | NO_STEP3_NEEDED | \['maciqwl3', '7qqemicc', 'rly4yj5j'\] → \['maciqwl3', '7qqemicc', 'rly4yj5j'\] | \[\]           | \[\]           | 864→864   |
| SCDSC         | mouse_kidney_10x   | NEEDS_STEP3     | \['1jpimoks', 'xb785wf2', 'z6sa8hrr'\] → \['1jpimoks', 'z6sa8hrr', 'hpgr57om'\] | \['xb785wf2'\] | \['hpgr57om'\] | 864→576   |
| SCTAG         | mouse_lung_cell    | NO_STEP3_NEEDED | \['dj8caoww', '3u7gzabk', 'qn5e9zad'\] → \['dj8caoww', '3u7gzabk', 'qn5e9zad'\] | \[\]           | \[\]           | 864→864   |
| SCTAG         | mouse_kidney_10x   | NEEDS_STEP3     | \['vufr5kzx', 'c56mch2v', 'gm79co9o'\] → \['vufr5kzx', 'c56mch2v', 'xknfppte'\] | \['gm79co9o'\] | \['xknfppte'\] | 864→576   |

逐名次 ID、acc、retention、顺延候选原始排名在 top3_changes.csv；完整候选表为每个任务 results/pipeline/best_test_acc.with_cell_retention.csv。SCDCC/kidney_10x 有低保留率候选但不在历史 Top3，因此不运行。

## 运行与完整性

| method        | dataset            | 状态            | 计划/启动/完成/有效 | guard skip | Traceback | Step3 sweeps                           | 新联合最优值 |
| ------------- | ------------------ | --------------- | ------------------- | ---------- | --------- | -------------------------------------- | ------------ |
| Graphsc       | mouse_bladder_cell | NO_STEP3_NEEDED | 0/0/0/0             | 0          | 0         | \[\]                                   | None         |
| Graphsc       | mouse_ES_cell      | NO_STEP3_NEEDED | 0/0/0/0             | 0          | 0         | \[\]                                   | None         |
| Graphsc       | mouse_lung_cell    | NO_STEP3_NEEDED | 0/0/0/0             | 0          | 0         | \[\]                                   | None         |
| Graphsc       | mouse_kidney_10x   | RUNNING_PILOT   | 30/23/13/13         | 0          | 13        | \['pumyvht5', 'zqmoa6w2', '8rot060a'\] | None         |
| SCDCC         | worm_neuron_cell   | NO_STEP3_NEEDED | 0/0/0/0             | 0          | 0         | \[\]                                   | None         |
| SCDCC         | mouse_bladder_cell | NO_STEP3_NEEDED | 0/0/0/0             | 0          | 0         | \[\]                                   | None         |
| SCDCC         | mouse_ES_cell      | NO_STEP3_NEEDED | 0/0/0/0             | 0          | 0         | \[\]                                   | None         |
| SCDCC         | mouse_lung_cell    | NO_STEP3_NEEDED | 0/0/0/0             | 0          | 0         | \[\]                                   | None         |
| SCDCC         | mouse_kidney_10x   | NO_STEP3_NEEDED | 0/0/0/0             | 0          | 0         | \[\]                                   | None         |
| SCDEEPCLUSTER | mouse_lung_cell    | NO_STEP3_NEEDED | 0/0/0/0             | 0          | 0         | \[\]                                   | None         |
| SCDEEPCLUSTER | mouse_kidney_10x   | PENDING         | 60/0/0/0            | 0          | 0         | \[\]                                   | None         |
| SCDSC         | mouse_bladder_cell | NO_STEP3_NEEDED | 0/0/0/0             | 0          | 0         | \[\]                                   | None         |
| SCDSC         | mouse_ES_cell      | NO_STEP3_NEEDED | 0/0/0/0             | 0          | 0         | \[\]                                   | None         |
| SCDSC         | mouse_lung_cell    | NO_STEP3_NEEDED | 0/0/0/0             | 0          | 0         | \[\]                                   | None         |
| SCDSC         | mouse_kidney_10x   | PENDING         | 60/0/0/0            | 0          | 0         | \[\]                                   | None         |
| SCTAG         | mouse_lung_cell    | NO_STEP3_NEEDED | 0/0/0/0             | 0          | 0         | \[\]                                   | None         |
| SCTAG         | mouse_kidney_10x   | PENDING         | 60/0/0/0            | 0          | 0         | \[\]                                   | None         |

新联合最优值为空的任务不允许更新最终成绩。old_joint_best 是历史 params CSV 中有限非负 acc 的最大值，仅作历史参考，不代表历史已经应用 0.8。

## Pilot、证据和诊断

- Pilot 为 Graphsc/mouse_kidney_10x；同一时间仅运行本次一个任务，不占满所有 GPU。控制器使用 CUDA_VISIBLE_DEVICES=0，LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64。
- 用户 systemd 服务启用 linger，SSH/Codex 退出后可继续；单任务不自动重试，三个相同确定性异常且无有效结果时停止本次任务。
- 预检查验证了全部 12 个 Step3 配置与 guarded Top3 对应；实际运行后再次比较生成配置哈希、N→M和 sweep 类型。
- ScaleFeature 曾在 Git 98bf305 仅改名为 ColumnSumNormalize，已恢复旧名兼容；搜索范围从原历史配置恢复，mode/eps 未改变。修改前文件保存在归档根目录。

### Graphsc/mouse_kidney_10x

状态：RUNNING_PILOT；诊断：无。
日志：`/egr/research-dselab/dingjia5/zhongyu/dance/examples/tuning/cluster_graphsc/mouse_kidney_10x/out_step3_cell_guard_papermachine_corrected.log`。
count=0 跳过证据：True；N→M：True；三个配置匹配：True；Step3 类型验证：True。
归档：`/egr/research-dselab/dingjia5/zhongyu/dance/examples/tuning/cluster_papermachine_step3_archive/corrected_counts_20260905T184935Z/cluster_graphsc/mouse_kidney_10x`。
命令：`/egr/research-dselab/dingjia5/anaconda3/envs/dance/bin/python -u main.py --dataset mouse_kidney_10x --data_dir ../temp_data --count 0 --device cuda --sweep_id mtmwfmye --summary_file_path results/pipeline/best_test_acc.corrected_step2.csv --cell_count_summary_path ../cluster_cell_filtering_papermachine.csv --min_cell_retention 0.8 --auto_additional_sweep_ids`
Sweep 明细：`[{"sweep_id": "pumyvht5", "planned": 10, "started": 3, "completed": 0, "valid": 0, "failed": 3, "guard_skips": 0}, {"sweep_id": "zqmoa6w2", "planned": 10, "started": 10, "completed": 10, "valid": 10, "failed": 0, "guard_skips": 0}, {"sweep_id": "8rot060a", "planned": 10, "started": 10, "completed": 3, "valid": 3, "failed": 7, "guard_skips": 0}]`

### SCDEEPCLUSTER/mouse_kidney_10x

状态：PENDING；诊断：无。
日志：`/egr/research-dselab/dingjia5/zhongyu/dance/examples/tuning/cluster_scdeepcluster/mouse_kidney_10x/out_step3_cell_guard_papermachine_corrected.log`。
count=0 跳过证据：False；N→M：False；三个配置匹配：False；Step3 类型验证：False。
归档：`/egr/research-dselab/dingjia5/zhongyu/dance/examples/tuning/cluster_papermachine_step3_archive/corrected_counts_20260905T184935Z/cluster_scdeepcluster/mouse_kidney_10x`。
命令：`/egr/research-dselab/dingjia5/anaconda3/envs/dance/bin/python -u main.py --dataset mouse_kidney_10x --data_dir ../temp_data --count 0 --device cuda --sweep_id zxk1e4sr --summary_file_path results/pipeline/best_test_acc.corrected_step2.csv --cell_count_summary_path ../cluster_cell_filtering_papermachine.csv --min_cell_retention 0.8 --auto_additional_sweep_ids`
Sweep 明细：`[]`

### SCDSC/mouse_kidney_10x

状态：PENDING；诊断：无。
日志：`/egr/research-dselab/dingjia5/zhongyu/dance/examples/tuning/cluster_scdsc/mouse_kidney_10x/out_step3_cell_guard_papermachine_corrected.log`。
count=0 跳过证据：False；N→M：False；三个配置匹配：False；Step3 类型验证：False。
归档：`/egr/research-dselab/dingjia5/zhongyu/dance/examples/tuning/cluster_papermachine_step3_archive/corrected_counts_20260905T184935Z/cluster_scdsc/mouse_kidney_10x`。
命令：`/egr/research-dselab/dingjia5/anaconda3/envs/dance/bin/python -u main.py --dataset mouse_kidney_10x --data_dir ../temp_data --count 0 --device cuda --sweep_id 5k60wsx4 --summary_file_path results/pipeline/best_test_acc.corrected_step2.csv --cell_count_summary_path ../cluster_cell_filtering_papermachine.csv --min_cell_retention 0.8 --auto_additional_sweep_ids`
Sweep 明细：`[]`

### SCTAG/mouse_kidney_10x

状态：PENDING；诊断：无。
日志：`/egr/research-dselab/dingjia5/zhongyu/dance/examples/tuning/cluster_sctag/mouse_kidney_10x/out_step3_cell_guard_papermachine_corrected.log`。
count=0 跳过证据：False；N→M：False；三个配置匹配：False；Step3 类型验证：False。
归档：`/egr/research-dselab/dingjia5/zhongyu/dance/examples/tuning/cluster_papermachine_step3_archive/corrected_counts_20260905T184935Z/cluster_sctag/mouse_kidney_10x`。
命令：`/egr/research-dselab/dingjia5/anaconda3/envs/dance/bin/python -u main.py --dataset mouse_kidney_10x --data_dir ../temp_data --count 0 --device cuda --sweep_id gg19inbf --summary_file_path results/pipeline/best_test_acc.corrected_step2.csv --cell_count_summary_path ../cluster_cell_filtering_papermachine.csv --min_cell_retention 0.8 --auto_additional_sweep_ids`
Sweep 明细：`[]`

## 历史保存与产物

归档根目录：`/egr/research-dselab/dingjia5/zhongyu/dance/examples/tuning/cluster_papermachine_step3_archive/corrected_counts_20260905T184935Z`。只有 NEEDS_STEP3 任务复制归档 results、config_yamls、temp_data 内对应目录、日志、YAML、原始 API 结果和元数据、Top3 映射、命令及计数 CSV。每个 archive_manifest.sha256 在启动前验证。
本轮没有 reset、checkout、删除用户文件、commit 或 push。预检查只生成临时配置；历史配置在完整复制归档后才允许被新的 Step3 配置覆盖。

- `/egr/research-dselab/dingjia5/zhongyu/dance/examples/tuning/cluster_cell_filtering_papermachine.csv`
- `/egr/research-dselab/dingjia5/zhongyu/dance/examples/tuning/cluster_cell_filtering_papermachine_summary.csv`
- `/egr/research-dselab/dingjia5/zhongyu/dance/examples/tuning/cluster_cell_filtering_papermachine_audit.md`
- `/egr/research-dselab/dingjia5/zhongyu/dance/examples/tuning/cluster_cell_filtering_papermachine_top3_changes.csv`
- `/egr/research-dselab/dingjia5/zhongyu/dance/examples/tuning/cluster_cell_filtering_papermachine_step3_results.csv`
