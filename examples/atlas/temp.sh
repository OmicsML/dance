#!/bin/bash

# 定义数组
array=("Blood" "Brain" "Heart" "Intestine" "Kidney" "Lung" "Pancreas")
# array=("Lung")

# 循环数组并在后台运行 Python 脚本
for tissue in "${array[@]}"
do
    python get_result_web.py --tissue "$tissue"
    echo "启动处理 tissue: $tissue"
done

# 等待所有后台进程完成
wait

echo "所有 Python 脚本已执行完成"
