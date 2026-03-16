#!/bin/bash

# 设置 API Key (建议不要在脚本中硬编码，可以使用环境变量)
export OPENAI_API_KEY="sk-92265d8b2c044989b10ad59e3a27b56f"

# 定义所有需要运行的方法列表
ALL_METHODS=(
    "cta_scdeepsort"
    "cta_scheteronet"
    "domain_efnst"
    "domain_louvain"
    "domain_spagcn"
    "domain_stagate"
    "cta_scgat"
    "cta_scrgcl"
    "cta_graphcs"
    "domain_stlearn"
    "domain_spagra"
)

# 循环遍历每一个方法
for TASK_NAME in "${ALL_METHODS[@]}"; do
    echo "------------------------------------------------"
    echo "Starting task: $TASK_NAME"
    
    # 根据当前的 TASK_NAME 设置变量
    PROMPT_FILE="pseudocode_${TASK_NAME}_prompt.txt"
    OUTPUT_DIR="${TASK_NAME}_openevolve_output"
    
    # 导出环境变量
    export TASK_NAME
    export OPENEVOLVE_PROMPT=$PROMPT_FILE

    # 运行 Python 命令
    python ../openevolve-run.py "$PROMPT_FILE" evaluator_pseudocode.py \
      --config config_pseudocode.yaml \
      --output "$OUTPUT_DIR"
      
    echo "Finished task: $TASK_NAME"
done

echo "All tasks completed."


  # --iterations 20 \
