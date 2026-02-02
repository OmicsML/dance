"""
Evaluator for HuggingFace dataset-based prompt optimization.
"""

import re
import traceback
import yaml
import os
import time
from openai import OpenAI
from tqdm import tqdm
from datasets import load_dataset
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

method_name=os.environ['TASK_NAME']
# Read config.yaml to get model settings
with open(os.path.join(os.path.dirname(__file__), "config_pseudocode.yaml"), "r") as f:
    config = yaml.safe_load(f)

# Get model settings from config
llm_config = config.get("llm", {})
api_base = llm_config.get("api_base", "http://localhost:1234/v1")

# Handle both single model and model list configurations
models = llm_config.get("models", [])
if models:
    # Use first model from list
    TASK_MODEL_NAME = models[0].get("name", "default-model")
else:
    # Fallback to direct model specification
    TASK_MODEL_NAME = llm_config.get("primary_model", "default-model")

# Get evaluator settings
evaluator_config = config.get("evaluator", {})
MAX_RETRIES = evaluator_config.get("max_retries", 3)

# Get max_tokens from LLM config
MAX_TOKENS = llm_config.get("max_tokens", 16000)
print(f"Using max_tokens: {MAX_TOKENS}")

# Initialize OpenAI client once for all evaluations
test_model = OpenAI(base_url=api_base)
print(f"Initialized OpenAI client with model: {TASK_MODEL_NAME}")

# Determine which dataset to use based on the OPENEVOLVE_PROMPT environment variable
import sys

prompt_file = os.environ.get("OPENEVOLVE_PROMPT")
if not prompt_file:
    # Default to a generic dataset config if not using the wrapper script
    evaluator_dir = os.path.dirname(os.path.abspath(__file__))
    DATASET_CONFIG_PATH = os.path.join(evaluator_dir, "dataset_settings.yaml")
    print("Warning: OPENEVOLVE_PROMPT not set. Using default dataset_settings.yaml")
else:
    basename = os.path.basename(prompt_file)
    dataset_filename = basename.replace(f"_{method_name}_prompt.txt", "_prompt_dataset.yaml").replace(
        ".txt", "_dataset.yaml"
    )
    evaluator_dir = os.path.dirname(os.path.abspath(__file__))
    DATASET_CONFIG_PATH = os.path.join(evaluator_dir, dataset_filename)
    print(f"Dataset configuration: {dataset_filename}")


def calculate_prompt_features(prompt):
    """
    Calculate custom features for MAP-Elites

    IMPORTANT: Returns raw continuous values, not bin indices.
    The database handles all scaling and binning automatically.

    Returns:
        tuple: (prompt_length, reasoning_sophistication_score)
        - prompt_length: Actual character count
        - reasoning_sophistication_score: Continuous score 0.0-1.0
    """
    # Feature 1: Prompt length (raw character count)
    prompt_length = len(prompt)

    # Feature 2: Reasoning sophistication score (continuous 0.0-1.0)
    prompt_lower = prompt.lower()
    sophistication_score = 0.0

    # Base scoring
    if len(prompt) >= 100:
        sophistication_score += 0.1  # Has substantial content

    # Check for few-shot examples (high sophistication)
    has_example = (
        "example" in prompt_lower
        or prompt.count("####") >= 4
        or bool(re.search(r"problem:.*?solution:", prompt_lower, re.DOTALL))
    )

    # Check for Chain-of-Thought (CoT) indicators
    has_cot = (
        "step by step" in prompt_lower
        or "step-by-step" in prompt_lower
        or any(phrase in prompt_lower for phrase in ["think through", "reasoning", "explain your"])
        or bool(re.search(r"(first|then|next|finally)", prompt_lower))
    )

    # Check for directive language
    has_directive = "solve" in prompt_lower or "calculate" in prompt_lower

    # Check for strict language
    has_strict = "must" in prompt_lower or "exactly" in prompt_lower

    # Calculate sophistication score
    if has_example:
        sophistication_score += 0.6  # Few-shot examples are sophisticated
        if has_cot:
            sophistication_score += 0.3  # Few-shot + CoT is most sophisticated
        elif len(prompt) > 1500:
            sophistication_score += 0.2  # Extensive few-shot
        else:
            sophistication_score += 0.1  # Basic few-shot
    elif has_cot:
        sophistication_score += 0.4  # Chain-of-thought
        if has_strict:
            sophistication_score += 0.2  # Strict CoT
        elif len(prompt) > 500:
            sophistication_score += 0.15  # Detailed CoT
        else:
            sophistication_score += 0.1  # Basic CoT
    else:
        # Basic prompts
        if has_directive:
            sophistication_score += 0.2  # Direct instruction
        else:
            sophistication_score += 0.1  # Simple prompt

    # Ensure score is within 0.0-1.0 range
    sophistication_score = min(1.0, max(0.0, sophistication_score))

    return prompt_length, sophistication_score


def load_prompt_config(prompt_path):
    """Load the prompt from text file and dataset config from matching _dataset.yaml file."""
    # Load prompt from text file
    with open(prompt_path, "r") as f:
        prompt = f.read().strip()
    
    # Load the configuration (already determined from environment variable)
    if not os.path.exists(DATASET_CONFIG_PATH):
        raise FileNotFoundError(f"Dataset configuration not found: {DATASET_CONFIG_PATH}")

    with open(DATASET_CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    return config, prompt


def load_hf_dataset(config):
    """Load HuggingFace dataset based on configuration."""
    dataset_name = config["dataset_name"]
    dataset_config = config.get("dataset_config", None)
    split = config.get("split", "test")

    print(f"Loading dataset: {dataset_name}")

    
    streaming = config.get("streaming", False)

    # Try to load the specified split
    if dataset_config:
        dataset = load_dataset(
            dataset_name,
            dataset_config,
            split=split,
            streaming=streaming,
        )
    else:
        dataset = load_dataset(
            dataset_name, split=split, streaming=streaming
        )
    filtered_dataset = dataset.filter(lambda example: example['method'] == method_name)
    # Print dataset info
    if hasattr(filtered_dataset, "__len__"):
        print(f"Dataset loaded with {len(filtered_dataset)} examples")
    else:
        print(f"Dataset loaded (streaming mode)")

    return filtered_dataset

from deepeval.models import DeepEvalBaseLLM
from openai import OpenAI, AsyncOpenAI
class CustomQwenPlus(DeepEvalBaseLLM):
    def __init__(self):
        self.api_key = os.getenv("DASHSCOPE_API_KEY")
        self.api_key="sk-92265d8b2c044989b10ad59e3a27b56f"
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.model_name = "qwen-flash"

        # 初始化同步客户端 (用于 generate)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        
        # 初始化异步客户端 (用于 a_generate) - 这对 DeepEval 的性能至关重要
        self.async_client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        """
        同步生成方法
        """
        chat_completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0, # 评估时建议将温度设为 0 以保证结果一致性
        )
        return chat_completion.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        """
        异步生成方法 - DeepEval 必须实现此方法
        """
        chat_completion = await self.async_client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return chat_completion.choices[0].message.content

    def get_model_name(self):
        return self.model_name
model = CustomQwenPlus()
def evaluate_single_sample(model, code, prompt_template):
    """
    使用GEval指标评估单个伪代码样本的质量。

    Args:
        model: DeepEval模型实例
        code: 原始代码
        prompt_template: 生成的伪代码

    Returns:
        float: 评估指标的平均得分
    """
    # 1. Readability (可读性)
    readability_metric = GEval(
         model=model,
        name="Readability",
        criteria="Readability - Variable names should be clear and the logic should be easy to follow.",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT], # 只需看生成的伪代码
    )

    # 2. Correctness (正确性)
    correctness_metric = GEval(
         model=model,
        name="Correctness",
        criteria="Correctness - The pseudo-code must remain faithful to the original code's logic. It should not alter the algorithm's intent.",
        # 需要对比"实际输出"和"预期输出/源代码"
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT]
    )

    # 3. Completeness (完整性)
    completeness_metric = GEval(
         model=model,
        name="Completeness",
        criteria="Completeness - The pseudo-code must cover all key boundary conditions and logical branches present in the original code.",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT]
    )

    # 4. Conciseness (简洁性)
    conciseness_metric = GEval(
         model=model,
        name="Conciseness",
        criteria="Conciseness - The pseudo-code should filter out unnecessary implementation details (like syntax-specific boilerplate) while keeping the core logic.",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT]
    )

    # 5. Maintainability (可维护性)
    maintainability_metric = GEval(
         model=model,
        name="Maintainability",
        criteria="Maintainability - The structure should be modular. Complex logic should be broken down into clear steps or blocks.",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT]
    )

    # 创建 DeepEval 测试用例
    test_case = LLMTestCase(
        input=code,                # 输入是原代码
        actual_output=prompt_template,# 模型生成的伪代码
        expected_output=code      # 这里的预期输出也是原代码（用于作为对比基准）
    )

    # ==========================================
    # 运行评估
    # ==========================================

    metrics = [
        readability_metric,
        correctness_metric,
        completeness_metric,
        conciseness_metric,
        maintainability_metric
    ]

    # 遍历运行每个指标
    print("开始评估...\n")
    accuracy = 0
    for metric in metrics:
        metric.measure(test_case)
        print(f"指标: {metric.name}")
        print(f"得分: {metric.score}")
        print(f"理由: {metric.reason}") # DeepEval 会生成打分理由，非常有价值
        print("-" * 30)
        accuracy += metric.score
    accuracy /= len(metrics)

    return accuracy

def evaluate_prompt(prompt, dataset, config, num_samples):
    """Evaluate a prompt on a subset of the dataset."""
    input_field = config["input_field"]
    target_field = config["target_field"]

    # Check dataset type

    # Sample from dataset - handle both streaming and non-streaming
    if hasattr(dataset, "take"):
        # Streaming dataset
        samples = dataset.take(num_samples)
        sample_iter = tqdm(samples, desc=f"Evaluating {num_samples} samples", total=num_samples)
    else:
        # Non-streaming dataset
        indices = range(min(num_samples, len(dataset)))
        samples = dataset.select(indices)
        sample_iter = tqdm(samples, desc=f"Evaluating {num_samples} samples")
    formatted_prompt = prompt.split('#Pseudocode-START')[1].split('#Pseudocode-END')[0]
    for example in sample_iter:
        input_text = example[input_field]
        expected = example[target_field]

        # Prepare the message for the LLM
        total_accuracy = 0
        num_evaluated_samples = 0
        # Call the LLM with retry logic
        for attempt in range(MAX_RETRIES):
            try:
                # Use max_tokens from config
                sample_accuracy = evaluate_single_sample(model, input_text, formatted_prompt)
                total_accuracy += sample_accuracy
                num_evaluated_samples += 1
                break
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    print(f"Failed to get response after {MAX_RETRIES} attempts: {e}")
                    raise e
                time.sleep(1)

    accuracy = total_accuracy / num_evaluated_samples if num_evaluated_samples > 0 else 0.0
    return accuracy


def evaluate_stage1(prompt_path):
    """
    Stage 1 evaluation: Quick evaluation with 10% of samples

    Args:
        prompt_path: Path to the prompt file

    Returns:
        Dictionary with combined_score metric
    """
    print("-" * 80)
    print("Starting Stage 1 evaluation...")
    print("-" * 80)

    try:
        # Load prompt configuration
        config, prompt = load_prompt_config(prompt_path)
        print(f"Loaded prompt configuration")

        # Load dataset
        dataset = load_hf_dataset(config)

        # Get number of samples from config
        num_samples = config.get("max_samples", 50)
        # Fixed to 10 samples for Stage 1 (quick evaluation)
        stage1_samples = 2

        print(f"Stage 1: Evaluating {stage1_samples} samples...")

        # Run evaluation
        accuracy = evaluate_prompt(prompt, dataset, config, stage1_samples)

        print(f"Stage 1 accuracy: {accuracy:.3f}")
        print("-" * 80)

        # Calculate custom features
        prompt_length, reasoning_sophistication = calculate_prompt_features(prompt)
        print(
            f"Prompt features - Length: {prompt_length} chars, Reasoning sophistication: {reasoning_sophistication:.3f}"
        )

        return {
            "combined_score": accuracy,
            "prompt_length": prompt_length,
            "reasoning_strategy": reasoning_sophistication,
        }

    except Exception as e:
        print(f"Stage 1 evaluation failed: {str(e)}")
        traceback.print_exc()
        print("-" * 80)

        # Always return feature dimensions, even on failure
        try:
            # Try to calculate features from the failed prompt
            with open(prompt_path, "r") as f:
                failed_prompt = f.read().strip()
            prompt_length, reasoning_sophistication = calculate_prompt_features(failed_prompt)
        except:
            # Fallback values if prompt can't be read
            prompt_length, reasoning_sophistication = 0, 0.0

        return {
            "combined_score": 0.0,
            "prompt_length": prompt_length,
            "reasoning_strategy": reasoning_sophistication,
            "error": str(e),
        }


def evaluate_stage2(prompt_path):
    """
    Stage 2 evaluation: Full evaluation with all samples

    Args:
        prompt_path: Path to the prompt file

    Returns:
        Dictionary with combined_score metric
    """
    print("-" * 80)
    print("Starting Stage 2 evaluation...")
    print("-" * 80)

    try:
        # Load prompt configuration
        config, prompt = load_prompt_config(prompt_path)
        print(f"Loaded prompt configuration")

        # Load dataset
        dataset = load_hf_dataset(config)

        # Get number of samples from config
        num_samples = config.get("max_samples", 50)
        # Fixed to 40 samples for Stage 2 (comprehensive evaluation)
        stage2_samples = 2

        print(f"Stage 2: Evaluating {stage2_samples} samples...")

        # Run evaluation
        accuracy = evaluate_prompt(prompt, dataset, config, stage2_samples)

        print(f"Stage 2 accuracy: {accuracy:.3f}")
        print("-" * 80)

        # Calculate custom features
        prompt_length, reasoning_sophistication = calculate_prompt_features(prompt)
        print(
            f"Prompt features - Length: {prompt_length} chars, Reasoning sophistication: {reasoning_sophistication:.3f}"
        )

        return {
            "combined_score": accuracy,
            "prompt_length": prompt_length,
            "reasoning_strategy": reasoning_sophistication,
        }

    except Exception as e:
        print(f"Stage 2 evaluation failed: {str(e)}")
        traceback.print_exc()
        print("-" * 80)

        # Always return feature dimensions, even on failure
        try:
            # Try to calculate features from the failed prompt
            with open(prompt_path, "r") as f:
                failed_prompt = f.read().strip()
            prompt_length, reasoning_sophistication = calculate_prompt_features(failed_prompt)
        except:
            # Fallback values if prompt can't be read
            prompt_length, reasoning_sophistication = 0, 0.0

        return {
            "combined_score": 0.0,
            "prompt_length": prompt_length,
            "reasoning_strategy": reasoning_sophistication,
            "error": str(e),
        }


def evaluate(prompt_path):
    """
    Main evaluation function - for backwards compatibility
    Calls evaluate_stage2 for full evaluation

    Args:
        prompt_path: Path to the prompt file

    Returns:
        Dictionary with combined_score metric
    """
    return evaluate_stage2(prompt_path)
