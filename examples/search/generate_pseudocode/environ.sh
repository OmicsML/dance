
export TASK_NAME="cta_scheteronet"
export OPENAI_API_KEY=sk-92265d8b2c044989b10ad59e3a27b56f
PROMPT_FILE="pseudocode_${TASK_NAME}_prompt.txt"
OUTPUT_DIR="${TASK_NAME}_openevolve_output"

export OPENEVOLVE_PROMPT=$PROMPT_FILE

python ../openevolve-run.py "$PROMPT_FILE" evaluator_pseudocode.py \
  --config config_pseudocode.yaml \
  --iterations 5 \
  --output "$OUTPUT_DIR"