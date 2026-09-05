# papermachine 聚类细胞过滤与历史 step3 审计

审计时间（UTC）：2026-09-05T07:03:50.079576+00:00。
源文件：`/egr/research-dselab/dingjia5/zhongyu/dance/examples/tuning/cluster_cell_filtering_papermachine.csv`；SHA256：`3ab6081bad4f835970c8bbb9d2ce1f5df5f52e724412131dbea01186f97d986e`。
范围依据源 CSV 与服务器分配表：5 个方法、5 个数据集、17 个 method/dataset；不是所有服务器的完整数据集集合。

## 结论

- 原 CSV 的 17 行全部是 MISSING_DATA 占位行，并非 17 个有效预处理场景。每组 YAML 有 6 种基因/细胞过滤组合，共 102 种；0 种有可验证的细胞计数，102 种保留率未知。
- 哪些 method/dataset 有低于 0.8 的组合：17 组全部无法判断。未确认任何一组低于阈值，也不能认定任何一组全部达标。
- 若启用 0.8，数学上应排除 kept_cells/raw_cells \< 0.8 的组合，恰好 0.8 应保留；当前无法列出实际应排除的组合。
- 5 个方法均归入第二类：当前参数存在，但没有历史传入并生效的证据。已核对的命令和日志没有该阈值；不能由当前 argparse 定义推出历史启用，也不能扩大为所有历史运行从未使用。
- 未启动训练、下载或移动 H5，未修改源 CSV、历史结果、日志或本地/远程 W&B 数据。

## 数据来源与维度可信度

每个源 CSV 的 h5 字段均为 MISSING，没有实际 H5 路径。逐一复查下述四级候选均不存在；raw_cells 为 NA，raw_genes 为空。汇总 CSV 将两者统一记为 NA。
同一 dataset 的维度一致性无法审查：没有可比较的原始维度；不是“已证明没有冲突”。不能从日志中的加工后矩阵形状倒推原始维度。
各数据集子目录没有 main.py，实际入口在对应 cluster\_\*/main.py。5 个入口默认 data_dir=../temp_data，ClusteringDataset 直接拼接 data_dir/<dataset>.h5。YAML 定义过滤参数，没有提供本次缺失 H5 的来源。
启动脚本使用 python main.py 和相对日志路径，支持从方法目录运行的解释；但历史元数据没有 cwd，root 为仓库根目录而不是可证明的工作目录。因此绝对历史路径仍标为无法确认。
从方法目录运行时，../temp_data 对应 `/egr/research-dselab/dingjia5/zhongyu/dance/examples/tuning/temp_data`（目前目录不存在）；./temp_data、./data 分别对应方法目录下的同名目录。

| method        | dataset            | raw_cells | raw_genes | H5/维度状态        |
| ------------- | ------------------ | --------- | --------- | ------------------ |
| Graphsc       | mouse_bladder_cell | NA        | NA        | MISSING / 无法核实 |
| Graphsc       | mouse_ES_cell      | NA        | NA        | MISSING / 无法核实 |
| Graphsc       | mouse_lung_cell    | NA        | NA        | MISSING / 无法核实 |
| Graphsc       | mouse_kidney_10x   | NA        | NA        | MISSING / 无法核实 |
| SCDCC         | worm_neuron_cell   | NA        | NA        | MISSING / 无法核实 |
| SCDCC         | mouse_bladder_cell | NA        | NA        | MISSING / 无法核实 |
| SCDCC         | mouse_ES_cell      | NA        | NA        | MISSING / 无法核实 |
| SCDCC         | mouse_lung_cell    | NA        | NA        | MISSING / 无法核实 |
| SCDCC         | mouse_kidney_10x   | NA        | NA        | MISSING / 无法核实 |
| SCDEEPCLUSTER | mouse_lung_cell    | NA        | NA        | MISSING / 无法核实 |
| SCDEEPCLUSTER | mouse_kidney_10x   | NA        | NA        | MISSING / 无法核实 |
| SCDSC         | mouse_bladder_cell | NA        | NA        | MISSING / 无法核实 |
| SCDSC         | mouse_ES_cell      | NA        | NA        | MISSING / 无法核实 |
| SCDSC         | mouse_lung_cell    | NA        | NA        | MISSING / 无法核实 |
| SCDSC         | mouse_kidney_10x   | NA        | NA        | MISSING / 无法核实 |
| SCTAG         | mouse_lung_cell    | NA        | NA        | MISSING / 无法核实 |
| SCTAG         | mouse_kidney_10x   | NA        | NA        | MISSING / 无法核实 |

四级候选路径逐项检查（全部不存在；不是实际读取路径）：

- **Graphsc/mouse_bladder_cell**：
  1. `/egr/research-dselab/dingjia5/zhongyu/dance/examples/tuning/temp_data/mouse_bladder_cell.h5` — 不存在
  1. `/egr/research-dselab/dingjia5/zhongyu/dance/examples/tuning/cluster_graphsc/temp_data/mouse_bladder_cell.h5` — 不存在
  1. `/egr/research-dselab/dingjia5/zhongyu/dance/examples/tuning/cluster_graphsc/data/mouse_bladder_cell.h5` — 不存在
  1. `/egr/research-dselab/dingjia5/zhongyu/dance/examples/single_modality/clustering/data/mouse_bladder_cell.h5` — 不存在
- **Graphsc/mouse_ES_cell**：
  1. `/egr/research-dselab/dingjia5/zhongyu/dance/examples/tuning/temp_data/mouse_ES_cell.h5` — 不存在
  1. `/egr/research-dselab/dingjia5/zhongyu/dance/examples/tuning/cluster_graphsc/temp_data/mouse_ES_cell.h5` — 不存在
  1. `/egr/research-dselab/dingjia5/zhongyu/dance/examples/tuning/cluster_graphsc/data/mouse_ES_cell.h5` — 不存在
  1. `/egr/research-dselab/dingjia5/zhongyu/dance/examples/single_modality/clustering/data/mouse_ES_cell.h5` — 不存在
- **Graphsc/mouse_lung_cell**：
  1. `/egr/research-dselab/dingjia5/zhongyu/dance/examples/tuning/temp_data/mouse_lung_cell.h5` — 不存在
  1. `/egr/research-dselab/dingjia5/zhongyu/dance/examples/tuning/cluster_graphsc/temp_data/mouse_lung_cell.h5` — 不存在
  1. `/egr/research-dselab/dingjia5/zhongyu/dance/examples/tuning/cluster_graphsc/data/mouse_lung_cell.h5` — 不存在
  1. `/egr/research-dselab/dingjia5/zhongyu/dance/examples/single_modality/clustering/data/mouse_lung_cell.h5` — 不存在
- **Graphsc/mouse_kidney_10x**：
  1. `/egr/research-dselab/dingjia5/zhongyu/dance/examples/tuning/temp_data/mouse_kidney_10x.h5` — 不存在
  1. `/egr/research-dselab/dingjia5/zhongyu/dance/examples/tuning/cluster_graphsc/temp_data/mouse_kidney_10x.h5` — 不存在
  1. `/egr/research-dselab/dingjia5/zhongyu/dance/examples/tuning/cluster_graphsc/data/mouse_kidney_10x.h5` — 不存在
  1. `/egr/research-dselab/dingjia5/zhongyu/dance/examples/single_modality/clustering/data/mouse_kidney_10x.h5` — 不存在
- **SCDCC/worm_neuron_cell**：
  1. `/egr/research-dselab/dingjia5/zhongyu/dance/examples/tuning/temp_data/worm_neuron_cell.h5` — 不存在
  1. `/egr/research-dselab/dingjia5/zhongyu/dance/examples/tuning/cluster_scdcc/temp_data/worm_neuron_cell.h5` — 不存在
  1. `/egr/research-dselab/dingjia5/zhongyu/dance/examples/tuning/cluster_scdcc/data/worm_neuron_cell.h5` — 不存在
  1. `/egr/research-dselab/dingjia5/zhongyu/dance/examples/single_modality/clustering/data/worm_neuron_cell.h5` — 不存在
- **SCDCC/mouse_bladder_cell**：
  1. `/egr/research-dselab/dingjia5/zhongyu/dance/examples/tuning/temp_data/mouse_bladder_cell.h5` — 不存在
  1. `/egr/research-dselab/dingjia5/zhongyu/dance/examples/tuning/cluster_scdcc/temp_data/mouse_bladder_cell.h5` — 不存在
  1. `/egr/research-dselab/dingjia5/zhongyu/dance/examples/tuning/cluster_scdcc/data/mouse_bladder_cell.h5` — 不存在
  1. `/egr/research-dselab/dingjia5/zhongyu/dance/examples/single_modality/clustering/data/mouse_bladder_cell.h5` — 不存在
- **SCDCC/mouse_ES_cell**：
  1. `/egr/research-dselab/dingjia5/zhongyu/dance/examples/tuning/temp_data/mouse_ES_cell.h5` — 不存在
  1. `/egr/research-dselab/dingjia5/zhongyu/dance/examples/tuning/cluster_scdcc/temp_data/mouse_ES_cell.h5` — 不存在
  1. `/egr/research-dselab/dingjia5/zhongyu/dance/examples/tuning/cluster_scdcc/data/mouse_ES_cell.h5` — 不存在
  1. `/egr/research-dselab/dingjia5/zhongyu/dance/examples/single_modality/clustering/data/mouse_ES_cell.h5` — 不存在
- **SCDCC/mouse_lung_cell**：
  1. `/egr/research-dselab/dingjia5/zhongyu/dance/examples/tuning/temp_data/mouse_lung_cell.h5` — 不存在
  1. `/egr/research-dselab/dingjia5/zhongyu/dance/examples/tuning/cluster_scdcc/temp_data/mouse_lung_cell.h5` — 不存在
  1. `/egr/research-dselab/dingjia5/zhongyu/dance/examples/tuning/cluster_scdcc/data/mouse_lung_cell.h5` — 不存在
  1. `/egr/research-dselab/dingjia5/zhongyu/dance/examples/single_modality/clustering/data/mouse_lung_cell.h5` — 不存在
- **SCDCC/mouse_kidney_10x**：
  1. `/egr/research-dselab/dingjia5/zhongyu/dance/examples/tuning/temp_data/mouse_kidney_10x.h5` — 不存在
  1. `/egr/research-dselab/dingjia5/zhongyu/dance/examples/tuning/cluster_scdcc/temp_data/mouse_kidney_10x.h5` — 不存在
  1. `/egr/research-dselab/dingjia5/zhongyu/dance/examples/tuning/cluster_scdcc/data/mouse_kidney_10x.h5` — 不存在
  1. `/egr/research-dselab/dingjia5/zhongyu/dance/examples/single_modality/clustering/data/mouse_kidney_10x.h5` — 不存在
- **SCDEEPCLUSTER/mouse_lung_cell**：
  1. `/egr/research-dselab/dingjia5/zhongyu/dance/examples/tuning/temp_data/mouse_lung_cell.h5` — 不存在
  1. `/egr/research-dselab/dingjia5/zhongyu/dance/examples/tuning/cluster_scdeepcluster/temp_data/mouse_lung_cell.h5` — 不存在
  1. `/egr/research-dselab/dingjia5/zhongyu/dance/examples/tuning/cluster_scdeepcluster/data/mouse_lung_cell.h5` — 不存在
  1. `/egr/research-dselab/dingjia5/zhongyu/dance/examples/single_modality/clustering/data/mouse_lung_cell.h5` — 不存在
- **SCDEEPCLUSTER/mouse_kidney_10x**：
  1. `/egr/research-dselab/dingjia5/zhongyu/dance/examples/tuning/temp_data/mouse_kidney_10x.h5` — 不存在
  1. `/egr/research-dselab/dingjia5/zhongyu/dance/examples/tuning/cluster_scdeepcluster/temp_data/mouse_kidney_10x.h5` — 不存在
  1. `/egr/research-dselab/dingjia5/zhongyu/dance/examples/tuning/cluster_scdeepcluster/data/mouse_kidney_10x.h5` — 不存在
  1. `/egr/research-dselab/dingjia5/zhongyu/dance/examples/single_modality/clustering/data/mouse_kidney_10x.h5` — 不存在
- **SCDSC/mouse_bladder_cell**：
  1. `/egr/research-dselab/dingjia5/zhongyu/dance/examples/tuning/temp_data/mouse_bladder_cell.h5` — 不存在
  1. `/egr/research-dselab/dingjia5/zhongyu/dance/examples/tuning/cluster_scdsc/temp_data/mouse_bladder_cell.h5` — 不存在
  1. `/egr/research-dselab/dingjia5/zhongyu/dance/examples/tuning/cluster_scdsc/data/mouse_bladder_cell.h5` — 不存在
  1. `/egr/research-dselab/dingjia5/zhongyu/dance/examples/single_modality/clustering/data/mouse_bladder_cell.h5` — 不存在
- **SCDSC/mouse_ES_cell**：
  1. `/egr/research-dselab/dingjia5/zhongyu/dance/examples/tuning/temp_data/mouse_ES_cell.h5` — 不存在
  1. `/egr/research-dselab/dingjia5/zhongyu/dance/examples/tuning/cluster_scdsc/temp_data/mouse_ES_cell.h5` — 不存在
  1. `/egr/research-dselab/dingjia5/zhongyu/dance/examples/tuning/cluster_scdsc/data/mouse_ES_cell.h5` — 不存在
  1. `/egr/research-dselab/dingjia5/zhongyu/dance/examples/single_modality/clustering/data/mouse_ES_cell.h5` — 不存在
- **SCDSC/mouse_lung_cell**：
  1. `/egr/research-dselab/dingjia5/zhongyu/dance/examples/tuning/temp_data/mouse_lung_cell.h5` — 不存在
  1. `/egr/research-dselab/dingjia5/zhongyu/dance/examples/tuning/cluster_scdsc/temp_data/mouse_lung_cell.h5` — 不存在
  1. `/egr/research-dselab/dingjia5/zhongyu/dance/examples/tuning/cluster_scdsc/data/mouse_lung_cell.h5` — 不存在
  1. `/egr/research-dselab/dingjia5/zhongyu/dance/examples/single_modality/clustering/data/mouse_lung_cell.h5` — 不存在
- **SCDSC/mouse_kidney_10x**：
  1. `/egr/research-dselab/dingjia5/zhongyu/dance/examples/tuning/temp_data/mouse_kidney_10x.h5` — 不存在
  1. `/egr/research-dselab/dingjia5/zhongyu/dance/examples/tuning/cluster_scdsc/temp_data/mouse_kidney_10x.h5` — 不存在
  1. `/egr/research-dselab/dingjia5/zhongyu/dance/examples/tuning/cluster_scdsc/data/mouse_kidney_10x.h5` — 不存在
  1. `/egr/research-dselab/dingjia5/zhongyu/dance/examples/single_modality/clustering/data/mouse_kidney_10x.h5` — 不存在
- **SCTAG/mouse_lung_cell**：
  1. `/egr/research-dselab/dingjia5/zhongyu/dance/examples/tuning/temp_data/mouse_lung_cell.h5` — 不存在
  1. `/egr/research-dselab/dingjia5/zhongyu/dance/examples/tuning/cluster_sctag/temp_data/mouse_lung_cell.h5` — 不存在
  1. `/egr/research-dselab/dingjia5/zhongyu/dance/examples/tuning/cluster_sctag/data/mouse_lung_cell.h5` — 不存在
  1. `/egr/research-dselab/dingjia5/zhongyu/dance/examples/single_modality/clustering/data/mouse_lung_cell.h5` — 不存在
- **SCTAG/mouse_kidney_10x**：
  1. `/egr/research-dselab/dingjia5/zhongyu/dance/examples/tuning/temp_data/mouse_kidney_10x.h5` — 不存在
  1. `/egr/research-dselab/dingjia5/zhongyu/dance/examples/tuning/cluster_sctag/temp_data/mouse_kidney_10x.h5` — 不存在
  1. `/egr/research-dselab/dingjia5/zhongyu/dance/examples/tuning/cluster_sctag/data/mouse_kidney_10x.h5` — 不存在
  1. `/egr/research-dselab/dingjia5/zhongyu/dance/examples/single_modality/clustering/data/mouse_kidney_10x.h5` — 不存在

## 预处理组合与保留率

`cell_retention = kept_cells / raw_cells`；仅当原始计数可靠且 raw_cells > 0 时计算。NA 不得当作 0 或达标。这里的 6 种只表示第一组基因过滤与细胞过滤的笛卡尔积，不是后续归一化、特征选择等完整流水线数量。

| gene_filter            | cell_filter            | scenario                                      | 实测保留率  |
| ---------------------- | ---------------------- | --------------------------------------------- | ----------- |
| FilterGenesPercentile  | FilterCellsScanpyOrder | FilterGenesPercentile+FilterCellsScanpyOrder  | NA：缺失 H5 |
| FilterGenesPercentile  | FilterCellsPlaceHolder | FilterGenesPercentile+FilterCellsPlaceHolder  | NA：缺失 H5 |
| FilterGenesScanpyOrder | FilterCellsScanpyOrder | FilterGenesScanpyOrder+FilterCellsScanpyOrder | NA：缺失 H5 |
| FilterGenesScanpyOrder | FilterCellsPlaceHolder | FilterGenesScanpyOrder+FilterCellsPlaceHolder | NA：缺失 H5 |
| FilterGenesPlaceHolder | FilterCellsScanpyOrder | FilterGenesPlaceHolder+FilterCellsScanpyOrder | NA：缺失 H5 |
| FilterGenesPlaceHolder | FilterCellsPlaceHolder | FilterGenesPlaceHolder+FilterCellsPlaceHolder | NA：缺失 H5 |

低于 0.8 的组合明细（无可判定记录；表为空不代表 0 个低保留率组合）：

| method | dataset | scenario | gene_filter | cell_filter | raw_cells | kept_cells | removed_cells | cell_retention | h5  | yaml |
| ------ | ------- | -------- | ----------- | ----------- | --------- | ---------- | ------------- | -------------- | --- | ---- |

按 method/dataset 汇总：

| method        | dataset            | 组合总数（YAML） | 实测组合数 | 低于0.8数量 | 最低保留率 | 最差组合 | 原始细胞数 | 最少保留细胞数 |
| ------------- | ------------------ | ---------------- | ---------- | ----------- | ---------- | -------- | ---------- | -------------- |
| Graphsc       | mouse_bladder_cell | 6                | 0          | NA          | NA         | NA       | NA         | NA             |
| Graphsc       | mouse_ES_cell      | 6                | 0          | NA          | NA         | NA       | NA         | NA             |
| Graphsc       | mouse_lung_cell    | 6                | 0          | NA          | NA         | NA       | NA         | NA             |
| Graphsc       | mouse_kidney_10x   | 6                | 0          | NA          | NA         | NA       | NA         | NA             |
| SCDCC         | worm_neuron_cell   | 6                | 0          | NA          | NA         | NA       | NA         | NA             |
| SCDCC         | mouse_bladder_cell | 6                | 0          | NA          | NA         | NA       | NA         | NA             |
| SCDCC         | mouse_ES_cell      | 6                | 0          | NA          | NA         | NA       | NA         | NA             |
| SCDCC         | mouse_lung_cell    | 6                | 0          | NA          | NA         | NA       | NA         | NA             |
| SCDCC         | mouse_kidney_10x   | 6                | 0          | NA          | NA         | NA       | NA         | NA             |
| SCDEEPCLUSTER | mouse_lung_cell    | 6                | 0          | NA          | NA         | NA       | NA         | NA             |
| SCDEEPCLUSTER | mouse_kidney_10x   | 6                | 0          | NA          | NA         | NA       | NA         | NA             |
| SCDSC         | mouse_bladder_cell | 6                | 0          | NA          | NA         | NA       | NA         | NA             |
| SCDSC         | mouse_ES_cell      | 6                | 0          | NA          | NA         | NA       | NA         | NA             |
| SCDSC         | mouse_lung_cell    | 6                | 0          | NA          | NA         | NA       | NA         | NA             |
| SCDSC         | mouse_kidney_10x   | 6                | 0          | NA          | NA         | NA       | NA         | NA             |
| SCTAG         | mouse_lung_cell    | 6                | 0          | NA          | NA         | NA       | NA         | NA             |
| SCTAG         | mouse_kidney_10x   | 6                | 0          | NA          | NA         | NA       | NA         | NA             |

## 历史 step3 阈值证据

分类：① 实际命令/配置明确启用 0.8 且日志证明筛选；② 当前参数存在，缺少历史启用证据；③ 有充分历史证据证明未使用（仅对可覆盖的运行成立）。未将“搜索没有命中”等同于第三类。

| method        | 分类 | 当前定义                                 | 实际历史证据结论                      |
| ------------- | ---- | ---------------------------------------- | ------------------------------------- |
| Graphsc       | ②    | cluster_graphsc/main.py，默认 None       | run.sh、所审日志和配置无阈值/筛选命中 |
| SCDCC         | ②    | cluster_scdcc/main.py，默认 None         | run.sh、所审日志和配置无阈值/筛选命中 |
| SCDEEPCLUSTER | ②    | cluster_scdeepcluster/main.py，默认 None | run.sh、所审日志和配置无阈值/筛选命中 |
| SCDSC         | ②    | cluster_scdsc/main.py，默认 None         | run.sh、所审日志和配置无阈值/筛选命中 |
| SCTAG         | ②    | cluster_sctag/main.py，默认 None         | run.sh、所审日志和配置无阈值/筛选命中 |

指定 rg 搜索正常完成，命中 35 行，全部位于 5 个 main.py。额外 --hidden --no-ignore 的全目录搜索被终止，其部分输出不能视为完整扫描。为此另完整读取 17 个 out.log、5 个 run.sh、所审数据集的 YAML，以及枚举到的全部 wandb-metadata.json 参数。未解析二进制 .wandb 文件，也未查询远程 W&B 或服务器 shell 历史。
专项读取：113 个 YAML；112 个历史结果 CSV 表头；元数据计数：{'cluster_graphsc': 3553}；元数据读取失败 0 个；实际参数中阈值命中 0 个。
元数据 host：\['papermachine'\]。本次枚举到的 3553 份元数据全部属于 Graphsc；其余方法依赖启动脚本、out.log 和历史 CSV，不能声称有同等元数据覆盖。
17 个 out.log 均存在，完整文本没有阈值及筛选/guard 关键字；所有已读结果 CSV 均无 preprocess.\* / cell_retention 列。现存 run.sh 是文件证据，不单独作为确已执行的证明。

| method        | dataset            | out.log 中实际 data_dir（首次行号）                                                                                                                      | 历史 params CSV 数 |
| ------------- | ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------ |
| Graphsc       | mouse_bladder_cell | ../temp_data @ examples/tuning/cluster_graphsc/mouse_bladder_cell/out.log:12                                                                             | 6                  |
| Graphsc       | mouse_ES_cell      | ../temp_data @ examples/tuning/cluster_graphsc/mouse_ES_cell/out.log:12                                                                                  | 6                  |
| Graphsc       | mouse_lung_cell    | ./temp_data @ examples/tuning/cluster_graphsc/mouse_lung_cell/out.log:9; ../temp_data @ examples/tuning/cluster_graphsc/mouse_lung_cell/out.log:100988   | 6                  |
| Graphsc       | mouse_kidney_10x   | ./temp_data @ examples/tuning/cluster_graphsc/mouse_kidney_10x/out.log:9; ../temp_data @ examples/tuning/cluster_graphsc/mouse_kidney_10x/out.log:106564 | 6                  |
| SCDCC         | worm_neuron_cell   | ../temp_data @ examples/tuning/cluster_scdcc/worm_neuron_cell/out.log:11                                                                                 | 6                  |
| SCDCC         | mouse_bladder_cell | ../temp_data @ examples/tuning/cluster_scdcc/mouse_bladder_cell/out.log:11                                                                               | 5                  |
| SCDCC         | mouse_ES_cell      | ../temp_data @ examples/tuning/cluster_scdcc/mouse_ES_cell/out.log:11                                                                                    | 6                  |
| SCDCC         | mouse_lung_cell    | ../temp_data @ examples/tuning/cluster_scdcc/mouse_lung_cell/out.log:11                                                                                  | 3                  |
| SCDCC         | mouse_kidney_10x   | ../temp_data @ examples/tuning/cluster_scdcc/mouse_kidney_10x/out.log:11                                                                                 | 3                  |
| SCDEEPCLUSTER | mouse_lung_cell    | ../temp_data @ examples/tuning/cluster_scdeepcluster/mouse_lung_cell/out.log:9                                                                           | 6                  |
| SCDEEPCLUSTER | mouse_kidney_10x   | ../temp_data @ examples/tuning/cluster_scdeepcluster/mouse_kidney_10x/out.log:9                                                                          | 6                  |
| SCDSC         | mouse_bladder_cell | ../temp_data @ examples/tuning/cluster_scdsc/mouse_bladder_cell/out.log:9                                                                                | 6                  |
| SCDSC         | mouse_ES_cell      | ../temp_data @ examples/tuning/cluster_scdsc/mouse_ES_cell/out.log:9                                                                                     | 6                  |
| SCDSC         | mouse_lung_cell    | ../temp_data @ examples/tuning/cluster_scdsc/mouse_lung_cell/out.log:10                                                                                  | 6                  |
| SCDSC         | mouse_kidney_10x   | ./data @ examples/tuning/cluster_scdsc/mouse_kidney_10x/out.log:9; ../temp_data @ examples/tuning/cluster_scdsc/mouse_kidney_10x/out.log:121991          | 6                  |
| SCTAG         | mouse_lung_cell    | ./data @ examples/tuning/cluster_sctag/mouse_lung_cell/out.log:7; ../temp_data @ examples/tuning/cluster_sctag/mouse_lung_cell/out.log:121260            | 6                  |
| SCTAG         | mouse_kidney_10x   | ./data @ examples/tuning/cluster_sctag/mouse_kidney_10x/out.log:7; ../temp_data @ examples/tuning/cluster_sctag/mouse_kidney_10x/out.log:120041          | 6                  |

Graphsc 实际运行参数样例（host=papermachine；均无 --min_cell_retention）：

- `examples/tuning/cluster_graphsc/wandb/run-20250605_145358-9n83q03e/files/wandb-metadata.json`：`--dataset mouse_ES_cell --count 400 --device cpu --sweep_id issfk09v --additional_sweep_ids hc4vqeuu`；cwd=None；root=`/egr/research-dselab/dingjia5/zhongyu/dance`。
- `examples/tuning/cluster_graphsc/wandb/run-20250530_002258-orvlecgb/files/wandb-metadata.json`：`--dataset mouse_bladder_cell --count 400 --device cuda:1 --sweep_id a0uxf25z --additional_sweep_ids 6j7i1pq5`；cwd=None；root=`/egr/research-dselab/dingjia5/zhongyu/dance`。
- `examples/tuning/cluster_graphsc/wandb/run-20240316_024611-boerndxr/files/wandb-metadata.json`：`--dataset mouse_kidney_10x --count 400 --device cuda:2`；cwd=None；root=`/egr/research-dselab/dingjia5/zhongyu/dance`。
- `examples/tuning/cluster_graphsc/wandb/run-20240316_005629-9hgtht0s/files/wandb-metadata.json`：`--dataset mouse_lung_cell --count 400 --device cuda:5`；cwd=None；root=`/egr/research-dselab/dingjia5/zhongyu/dance`。

当前代码的生效条件（不是历史生效证明）：

- `dance/pipeline.py:1210`：仅 min_cell_retention 非 None 且存在 preprocess.cell_retention 列才筛选；`1214` 输出 Filtered step-2 pipelines；缺列在 `1217` 警告并跳过筛选。筛选后才取 top-k。
- `dance/pipeline.py:93` 的运行时 guard 只在阈值与统计值存在且低于阈值时跳过；5 个 main.py 均传递参数，但默认 None。
- `examples/tuning/generate_cluster_112_step3_commands.py` 可以生成带阈值的命令，但面向 112 的生成代码不是 papermachine 历史执行证明。
- 不应将当前全 MISSING_DATA 的 CSV 直接作为可信 cell_count_summary_path：缺失组合无法匹配，可能生成 NaN，比较筛选可能误删未知行。需先补齐可靠计数。

## 旧结果重算范围

为提供“保留率至少 0.8”的可比较结果，上表全部 17 组都需先恢复并核实实际原始数据版本，重算 102 种过滤场景的细胞计数，再按阈值重新筛选/排名 step2 候选并核对现存 step3 配置。现有 params 结果不能标注为“已受 0.8 约束”。
具体哪些旧聚类训练必须重跑尚无法确定：低于 0.8 的旧场景应退出受约束比较；重新筛选后新入选但无可复用结果的场景才需训练。若数据版本或实际流水线发生变化，相应结果也需重算。不能据缺失数据就断言全部历史训练必须重跑。
额外结果缺口：以下组合没有现存 results/pipeline/best_test_acc.csv：无。其它 params CSV 存在不能替代缺失的 step2 候选排名。

已审查的历史结果文件与计数列（均为缺失）：

- `examples/tuning/cluster_graphsc/mouse_bladder_cell/results/params/0_best.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_graphsc/mouse_bladder_cell/results/params/0_best_test_acc.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_graphsc/mouse_bladder_cell/results/params/1_best.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_graphsc/mouse_bladder_cell/results/params/1_best_test_acc.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_graphsc/mouse_bladder_cell/results/params/2_best.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_graphsc/mouse_bladder_cell/results/params/2_best_test_acc.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_graphsc/mouse_bladder_cell/results/pipeline/best_test_acc.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_graphsc/mouse_ES_cell/results/params/0_best.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_graphsc/mouse_ES_cell/results/params/0_best_test_acc.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_graphsc/mouse_ES_cell/results/params/1_best.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_graphsc/mouse_ES_cell/results/params/1_best_test_acc.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_graphsc/mouse_ES_cell/results/params/2_best.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_graphsc/mouse_ES_cell/results/params/2_best_test_acc.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_graphsc/mouse_ES_cell/results/pipeline/best_test_acc.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_graphsc/mouse_lung_cell/results/params/0_best.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_graphsc/mouse_lung_cell/results/params/0_best_test_acc.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_graphsc/mouse_lung_cell/results/params/1_best.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_graphsc/mouse_lung_cell/results/params/1_best_test_acc.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_graphsc/mouse_lung_cell/results/params/2_best.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_graphsc/mouse_lung_cell/results/params/2_best_test_acc.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_graphsc/mouse_lung_cell/results/pipeline/best_test_acc.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_graphsc/mouse_kidney_10x/results/params/0_best.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_graphsc/mouse_kidney_10x/results/params/0_best_test_acc.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_graphsc/mouse_kidney_10x/results/params/1_best.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_graphsc/mouse_kidney_10x/results/params/1_best_test_acc.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_graphsc/mouse_kidney_10x/results/params/2_best.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_graphsc/mouse_kidney_10x/results/params/2_best_test_acc.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_graphsc/mouse_kidney_10x/results/pipeline/best_test_acc.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_scdcc/worm_neuron_cell/results/params/0_best.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_scdcc/worm_neuron_cell/results/params/0_best_test_acc.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_scdcc/worm_neuron_cell/results/params/1_best.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_scdcc/worm_neuron_cell/results/params/1_best_test_acc.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_scdcc/worm_neuron_cell/results/params/2_best.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_scdcc/worm_neuron_cell/results/params/2_best_test_acc.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_scdcc/worm_neuron_cell/results/pipeline/best_test_acc.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_scdcc/mouse_bladder_cell/results/params/0_best.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_scdcc/mouse_bladder_cell/results/params/0_best_test_acc.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_scdcc/mouse_bladder_cell/results/params/1_best.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_scdcc/mouse_bladder_cell/results/params/1_best_test_acc.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_scdcc/mouse_bladder_cell/results/params/2_best.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_scdcc/mouse_bladder_cell/results/pipeline/best_test_acc.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_scdcc/mouse_ES_cell/results/params/0_best.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_scdcc/mouse_ES_cell/results/params/0_best_test_acc.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_scdcc/mouse_ES_cell/results/params/1_best.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_scdcc/mouse_ES_cell/results/params/1_best_test_acc.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_scdcc/mouse_ES_cell/results/params/2_best.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_scdcc/mouse_ES_cell/results/params/2_best_test_acc.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_scdcc/mouse_ES_cell/results/pipeline/best_test_acc.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_scdcc/mouse_lung_cell/results/params/0_best.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_scdcc/mouse_lung_cell/results/params/1_best.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_scdcc/mouse_lung_cell/results/params/2_best.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_scdcc/mouse_lung_cell/results/pipeline/best_test_acc.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_scdcc/mouse_kidney_10x/results/params/0_best.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_scdcc/mouse_kidney_10x/results/params/1_best.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_scdcc/mouse_kidney_10x/results/params/2_best.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_scdcc/mouse_kidney_10x/results/pipeline/best_test_acc.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_scdeepcluster/mouse_lung_cell/results/params/0_best.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_scdeepcluster/mouse_lung_cell/results/params/0_best_test_acc.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_scdeepcluster/mouse_lung_cell/results/params/1_best.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_scdeepcluster/mouse_lung_cell/results/params/1_best_test_acc.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_scdeepcluster/mouse_lung_cell/results/params/2_best.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_scdeepcluster/mouse_lung_cell/results/params/2_best_test_acc.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_scdeepcluster/mouse_lung_cell/results/pipeline/best_test_acc.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_scdeepcluster/mouse_kidney_10x/results/params/0_best.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_scdeepcluster/mouse_kidney_10x/results/params/0_best_test_acc.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_scdeepcluster/mouse_kidney_10x/results/params/1_best.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_scdeepcluster/mouse_kidney_10x/results/params/1_best_test_acc.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_scdeepcluster/mouse_kidney_10x/results/params/2_best.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_scdeepcluster/mouse_kidney_10x/results/params/2_best_test_acc.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_scdeepcluster/mouse_kidney_10x/results/pipeline/best_test_acc.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_scdsc/mouse_bladder_cell/results/params/0_best.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_scdsc/mouse_bladder_cell/results/params/0_best_test_acc.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_scdsc/mouse_bladder_cell/results/params/1_best.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_scdsc/mouse_bladder_cell/results/params/1_best_test_acc.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_scdsc/mouse_bladder_cell/results/params/2_best.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_scdsc/mouse_bladder_cell/results/params/2_best_test_acc.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_scdsc/mouse_bladder_cell/results/pipeline/best_test_acc.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_scdsc/mouse_ES_cell/results/params/0_best.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_scdsc/mouse_ES_cell/results/params/0_best_test_acc.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_scdsc/mouse_ES_cell/results/params/1_best.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_scdsc/mouse_ES_cell/results/params/1_best_test_acc.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_scdsc/mouse_ES_cell/results/params/2_best.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_scdsc/mouse_ES_cell/results/params/2_best_test_acc.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_scdsc/mouse_ES_cell/results/pipeline/best_test_acc.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_scdsc/mouse_lung_cell/results/params/0_best.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_scdsc/mouse_lung_cell/results/params/0_best_test_acc.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_scdsc/mouse_lung_cell/results/params/1_best.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_scdsc/mouse_lung_cell/results/params/1_best_test_acc.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_scdsc/mouse_lung_cell/results/params/2_best.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_scdsc/mouse_lung_cell/results/params/2_best_test_acc.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_scdsc/mouse_lung_cell/results/pipeline/best_test_acc.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_scdsc/mouse_kidney_10x/results/params/0_best.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_scdsc/mouse_kidney_10x/results/params/0_best_test_acc.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_scdsc/mouse_kidney_10x/results/params/1_best.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_scdsc/mouse_kidney_10x/results/params/1_best_test_acc.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_scdsc/mouse_kidney_10x/results/params/2_best.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_scdsc/mouse_kidney_10x/results/params/2_best_test_acc.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_scdsc/mouse_kidney_10x/results/pipeline/best_test_acc.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_sctag/mouse_lung_cell/results/params/0_best.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_sctag/mouse_lung_cell/results/params/0_best_test_acc.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_sctag/mouse_lung_cell/results/params/1_best.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_sctag/mouse_lung_cell/results/params/1_best_test_acc.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_sctag/mouse_lung_cell/results/params/2_best.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_sctag/mouse_lung_cell/results/params/2_best_test_acc.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_sctag/mouse_lung_cell/results/pipeline/best_test_acc.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_sctag/mouse_kidney_10x/results/params/0_best.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_sctag/mouse_kidney_10x/results/params/0_best_test_acc.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_sctag/mouse_kidney_10x/results/params/1_best.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_sctag/mouse_kidney_10x/results/params/1_best_test_acc.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_sctag/mouse_kidney_10x/results/params/2_best.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_sctag/mouse_kidney_10x/results/params/2_best_test_acc.csv`：无 preprocess/cell_retention 列。
- `examples/tuning/cluster_sctag/mouse_kidney_10x/results/pipeline/best_test_acc.csv`：无 preprocess/cell_retention 列。

## 产物与变更

- 汇总：`/egr/research-dselab/dingjia5/zhongyu/dance/examples/tuning/cluster_cell_filtering_papermachine_summary.csv`（17 条数据；NA 表示无法确定）。
- 本轮只新增 summary CSV 与 audit.md；analyze_cluster_cell_filtering.py 的 3 行删除是上一轮已有改动。
- CSV 受现有 Git 忽略规则影响，文件存在不保证出现在 git diff。源 CSV SHA256 在生成后复核不变。
