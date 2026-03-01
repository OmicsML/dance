#!/usr/bin/env python3
"""
Unified evaluation script for GEPA benchmark datasets.
Can evaluate baseline or evolved prompts on IFEval, HoVer, and HotpotQA.
"""

import os
import json
import yaml
import time
import argparse
from datetime import datetime
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams


# Initialize OpenAI client
def get_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    api_key="sk-92265d8b2c044989b10ad59e3a27b56f"
    return OpenAI(base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", api_key=api_key)


def load_prompt(dataset_name, method_name,prompt_type="baseline"):
    """Load prompt template for a dataset."""
    if prompt_type == "baseline":
        prompt_path = f"{dataset_name}_{method_name}_prompt.txt"
    else:  # evolved
        prompt_path = f"openevolve_output_qwen3_{dataset_name}/best/best_program.txt"

    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    with open(prompt_path, "r") as f:
        return f.read().strip()



import os
from openai import OpenAI, AsyncOpenAI
from deepeval.models import DeepEvalBaseLLM

class CustomQwenPlus(DeepEvalBaseLLM):
    def __init__(self,model_name="qwen-flash"):
        self.api_key = os.getenv("DASHSCOPE_API_KEY")
        self.api_key="sk-92265d8b2c044989b10ad59e3a27b56f"
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.model_name = model_name

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


def evaluate_pseudocode(client, prompt_template,num_samples,model,method_name):
    from string import Template

    """Evaluate pseudocode dataset."""
    print("\nLoading pseudocode dataset...")

    # Try test split first, then train
    split_used = "train"
    dataset = load_dataset("zhongyuxing/Graph_Structure_Learning_Pseudocode_new_new", split=split_used)
    dataset=dataset.filter(lambda example: example['method'] == method_name)
    # Determine samples to process
    samples_to_process = min(num_samples, len(dataset))
    print(f"Using full {split_used} split: {samples_to_process} samples")
    dataset_iter = tqdm(dataset, desc="Evaluating")
    prompt = prompt_template.split('#Pseudocode-START')[1].split('#Pseudocode-END')[0]
    total_accuracy = 0
    num_evaluated_samples = 0

    for i, example in enumerate(dataset_iter):
        if num_samples is not None and i >= samples_to_process:
            break
        code = example["code"]
        # 使用新的评估函数评估单个样本
        sample_accuracy = evaluate_single_sample(model, code, prompt)

        # 累加样本准确率
        total_accuracy += sample_accuracy
        num_evaluated_samples += 1

    # 计算平均准确率
    average_accuracy = total_accuracy / num_evaluated_samples if num_evaluated_samples > 0 else 0

    return average_accuracy

def main():
    parser = argparse.ArgumentParser(description="Evaluate prompts on GEPA benchmark datasets")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["pseudocode"],
        help="Dataset to evaluate on",
    )
    parser.add_argument(
        "--method_name",
        type=str,
        choices=["cta_scdeepsort", "cta_scheteronet", "domain_efnst", "domain_louvain","domain_spagcn","domain_stagate"],
        help="Method to generate",
        default="cta_scheteronet"
    )
    parser.add_argument(
        "--prompt-type",
        type=str,
        default="baseline",
        choices=["baseline", "evolved"],
        help="Type of prompt to use",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=2,
        help="Number of samples to evaluate (default: full dataset)",
    )
    parser.add_argument(
        "--model", type=str, default="qwen-flash", help="Model to use for evaluation"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output file for results (default: auto-generated)"
    )

    args = parser.parse_args()

    # Initialize client
    client = get_client()
    datasets = [args.dataset]

    # Evaluation functions
    eval_funcs = {"pseudocode":evaluate_pseudocode}


    # Store results
    all_results = []

    print(f"\n{'='*60}")
    print(f"PROMPT EVALUATION - {args.prompt_type.upper()}")
    print(f"Model: {args.model}")
    if args.samples:
        print(f"Samples per dataset: {args.samples}")
    else:
        print(f"Samples per dataset: Full dataset")
    print(f"{'='*60}")

    for dataset_name in datasets:
        print(f"\nEvaluating {dataset_name.upper()}...")
        test_model = CustomQwenPlus(args.model)
        try:
            # Load prompt
            prompt_template = load_prompt(dataset_name, args.method_name,args.prompt_type)
            print(f"Loaded {args.prompt_type} prompt ({len(prompt_template)} chars)")

            # Run evaluation
            start_time = time.time()
            accuracy = eval_funcs[dataset_name](
                client, prompt_template, args.samples, test_model,args.method_name
            )
            elapsed_time = time.time() - start_time


            # Store result
            result = {
                "dataset": dataset_name,
                "prompt_type": args.prompt_type,
                "accuracy": accuracy,
                "elapsed_time": elapsed_time,
                "timestamp": datetime.now().isoformat(),
            }

            all_results.append(result)

            # Print results
            print(f"\nResults for {dataset_name.upper()}:")
            print(f"  Accuracy: {accuracy:.3f})")
            print(f"  Time: {elapsed_time:.1f}s ({elapsed_time:.1f})")

        except Exception as e:
            print(f"Error evaluating {dataset_name}: {str(e)}")
            all_results.append(
                {
                    "dataset": dataset_name,
                    "prompt_type": args.prompt_type,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                }
            )

    # Save results
    output_path = args.output
    if not output_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"evaluation_results_{args.prompt_type}_{timestamp}.json"

    final_results = {
        "prompt_type": args.prompt_type,
        "model": args.model,
        "samples_per_dataset": args.samples,
        "timestamp": datetime.now().isoformat(),
        "results": all_results,
    }

    # Calculate aggregate statistics
    valid_results = [r for r in all_results if "error" not in r]
    if valid_results:
        total_accuracy = sum(r["accuracy"] for r in valid_results)
        aggregate_accuracy = total_accuracy / len(valid_results) if len(valid_results) > 0 else 0

        final_results["summary"] = {
            "aggregate_accuracy": aggregate_accuracy,
            "total_accuracy": total_accuracy,
            "datasets_evaluated": len(valid_results),
        }

    with open(output_path, "w") as f:
        json.dump(final_results, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")

    for result in all_results:
        if "error" not in result:
            print(f"\n{result['dataset'].upper()}:")
            print(f"  Accuracy: {result['accuracy']:.3f}")

    if "summary" in final_results:
        print(f"\nAGGREGATE:")
        print(f"  Overall Accuracy: {final_results['summary']['aggregate_accuracy']:.3f}")
        print(f"  Total Samples: {final_results['summary']['datasets_evaluated']}")

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
