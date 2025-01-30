#!/bin/bash

# 定义数组
array=("blood" "brain" "heart" "intestine" "kidney" "lung" "pancreas")
# 循环数组并在后台运行 Python 脚本
for tissue in "${array[@]}"
do
    # python similarity/example_usage_anndata.py --tissue "$tissue" >> example_usage_anndata.log 2>&1
    python similarity/optimize_similarity_weights.py --tissue "$tissue"
    python visualization/visualize_atlas_performance.py --tissue "$tissue"
    python similarity/optimize_similarity_weights.py --tissue "$tissue" --in_query
    python visualization/visualize_atlas_performance.py --tissue "$tissue" --in_query
    python similarity/optimize_similarity_weights.py --tissue "$tissue" --reduce_error
    python visualization/visualize_atlas_performance.py --tissue "$tissue" --reduce_error
    python similarity/optimize_similarity_weights.py --tissue "$tissue" --in_query --reduce_error
    python visualization/visualize_atlas_performance.py --tissue "$tissue" --in_query --reduce_error
    echo "启动处理 tissue: $tissue"
done

# 等待所有后台进程完成
wait

echo "所有 Python 脚本已执行完成"
