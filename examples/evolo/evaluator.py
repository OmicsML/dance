"""
Evaluator for the scDeepSort cell type annotation script.
"""

import json
import subprocess
import re
import time
import numpy as np
import os
import traceback
import ast  # 新增：导入 ast 模块以避免 literal_eval 报错
from openevolve.evaluation_result import EvaluationResult
import logging

from dance.modules.spatial.spatial_domain import spagcn
from dance.settings import EXAMPLESDIR

stage1_args=os.getenv("stage1_args")
BENCHMARKS_args=os.getenv("BENCHMARKS_args")
with open(f"{EXAMPLESDIR}/evolo/benchmarks_config.json", "r") as f:
    BENCHMARKS = json.load(f)
stage1_args = BENCHMARKS[stage1_args]
BENCHMARKS = BENCHMARKS[BENCHMARKS_args]

BENCHMARK_TIMEOUT = 600000  # 1000 minutes
logger = logging.getLogger(__name__)

def _run_benchmark(program_path, benchmark_name, benchmark_args, timeout):
    """Helper function to run a single benchmark command."""
    command = ["python", program_path, "--cache"] + benchmark_args
    
    start_time = time.time()
    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False  
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        if process.returncode != 0:
            return {
                "status": "error", "score": 0.0, "time": execution_time,
                "error": f"Process exited with code {process.returncode}",
                "stdout": process.stdout, "stderr": process.stderr,
            }

        output = process.stdout
        
        # 1. 增加对 times 列表的捕获
        scores_match = re.search(r"scores:\s*(\[.*?\])", output)
        inner_scores_match = re.search(r"inner_scores:\s*(\[.*?\])", output)
        times_match = re.search(r"times:\s*(\[.*?\])", output)

        # 2. 如果缺少任意一个，返回 error
        if not scores_match or not inner_scores_match or not times_match:
            return {
                "status": "error", "scores": [], "inner_scores": [], "times": [], "time": execution_time,
                "error": "Could not parse 'scores', 'inner_scores', or 'times' lists from output.",
                "stdout": output, "stderr": process.stderr,
            }

        try:
            # 3. 提取并转换字符串为 Python 列表
            scores_list = ast.literal_eval(scores_match.group(1))
            inner_scores_list = ast.literal_eval(inner_scores_match.group(1))
            times_list = ast.literal_eval(times_match.group(1))
            
        except (ValueError, SyntaxError) as e:
            return {
                "status": "error", "scores": [], "inner_scores": [], "times": [], "time": execution_time,
                "error": f"Error parsing list string to Python list: {str(e)}",
                "stdout": output, "stderr": process.stderr,
            }

        # 4. 成功时，将计算出的内部平均时间赋给 'time'
        internal_mean_time = np.mean(times_list) if times_list else execution_time
        
        return {
            "status": "success", 
            "scores": scores_list, 
            "inner_scores": inner_scores_list, 
            "times": times_list,
            "score": np.mean(scores_list),
            "inner_score": np.mean(inner_scores_list),
            "time": internal_mean_time,  # 使用内部时间
            "total_execution_time": execution_time, # 保留外部执行时间做备份参考
            "error": None, 
            "stdout": output, 
            "stderr": process.stderr,
        }

    except subprocess.TimeoutExpired as e:
        return {
            "status": "error", "score": 0.0, "time": timeout,
            "error": f"Timeout of {timeout} seconds exceeded.",
            "stdout": e.stdout if e.stdout else "", "stderr": e.stderr if e.stderr else "",
        }
    except Exception as e:
        return {
            "status": "error", "score": 0.0, "time": time.time() - start_time,
            "error": f"An unexpected error occurred: {str(e)}",
            "stdout": "", "stderr": traceback.format_exc(),
        }

def evaluate(program_path):
    results = {}
    successful_runs = []
    failed_runs = []
    
    # 只保留用于计算全局平均值的标量列表
    all_inner_scores = []
    all_speed_scores = []
    all_accuracy_scores = [] 
    all_times = [] 

    for name, args in BENCHMARKS.items():
        print(f"--- Running benchmark: {name} ---")
        result = _run_benchmark(program_path, name, args, BENCHMARK_TIMEOUT)
        results[name] = result
        
        if result["status"] == "success":
            successful_runs.append(result)
            
            i_score = result.get("inner_score", 0.0)
            e_score = result.get("score", 0.0)
            
            t = result.get("time", 300.0)
            s_score = 1.0 / (1.0 + t / 300.0)
            
            all_inner_scores.append(i_score)
            all_accuracy_scores.append(e_score)
            all_speed_scores.append(s_score)
            all_times.append(t)
            
            print(f"Success! Inner Score: {i_score:.4f}, Speed Score: {s_score:.4f}, Mean Run Time: {t:.2f}s")
            
        else:
            failed_runs.append(result)
            all_inner_scores.append(0.0)
            all_accuracy_scores.append(0.0)
            all_speed_scores.append(0.0)
            
            print(f"Failed! Error: {result['error']}")
            logger.info(f"Stdout: {result['stdout']}")
            logger.info(f"Stderr: {result['stderr']}")

    total_benchmarks = len(BENCHMARKS)
    if total_benchmarks == 0:
        return EvaluationResult(metrics={"combined_score": 0.0, "error": "No benchmarks defined."}, artifacts={})

    avg_inner_accuracy = np.mean(all_inner_scores) 
    avg_accuracy = np.mean(all_accuracy_scores)
    avg_speed_score = np.mean(all_speed_scores)
    reliability_score = len(successful_runs) / total_benchmarks
    avg_success_time = np.mean(all_times) if all_times else 0.0

    combined_score = (0.8 * avg_inner_accuracy) + (0.2 * avg_speed_score)

    artifacts = {
        "benchmark_summary": {
            name: {
                "status": res["status"], 
                "inner_score": round(res.get("inner_score", 0), 4), 
                "time": round(res.get("time", 0), 2),
                "error": res.get("error")
            }
            for name, res in results.items()
        },
        "performance_overview": (
            f"Global Score: {combined_score:.4f}. "
            f"Reliability: {len(successful_runs)}/{total_benchmarks}. "
            f"Avg Accuracy (Global): {avg_inner_accuracy:.4f}."
        )
    }

    if not successful_runs:
        artifacts["error_details"] = {
            "error_type": "AllBenchmarksFailed",
            "suggestion": "Check dependencies or syntax. All runs crashed or timed out.",
            "sample_stderr": failed_runs[0]['stderr'][-1000:] if failed_runs and failed_runs[0]['stderr'] else "N/A"
        }

    metrics = {
        "combined_score": float(combined_score),
        "avg_accuracy": float(avg_accuracy),
        "avg_inner_accuracy": float(avg_inner_accuracy), 
        "reliability_score": float(reliability_score),
        "avg_speed_score": float(avg_speed_score),       
        "avg_success_time": float(avg_success_time)
    }

    # 重点在这里：直接从结果字典 res 中提取标量和列表，统一放入 metrics 中
    for name, res in results.items():
        # 记录整体平均值和状态（标量）
        metrics[f"{name}_score"] = float(res.get("score", 0))
        metrics[f"{name}_inner_score"] = float(res.get("inner_score", 0))
        metrics[f"{name}_time"] = float(res.get("time", 0))
        metrics[f"{name}_status"] = 1.0 if res["status"] == "success" else 0.0
        
        # 提取列表数据，若失败或不存在则用 [] 兜底
        scores_list = res.get("scores", [])
        inner_scores_list = res.get("inner_scores", [])
        times_list = res.get("times", [])
        
        # 将列表展平为独立的标量 metric
        for i, val in enumerate(scores_list):
            metrics[f"{name}_score_{i}"] = float(val)
            
        for i, val in enumerate(inner_scores_list):
            metrics[f"{name}_inner_score_{i}"] = float(val)
            
        for i, val in enumerate(times_list):
            metrics[f"{name}_time_{i}"] = float(val)

    return EvaluationResult(
        metrics=metrics,
        artifacts=artifacts
    )
    
def evaluate_stage1(program_path):
    print("--- Running Stage 1 Evaluation ---")
    timeout = 240000 
    result = _run_benchmark(program_path, "Stage1_Check", stage1_args, timeout)
    if result["status"] == "success":
        return EvaluationResult(
            metrics={"combined_score": 1.0, "runs_successfully": 1.0},
            artifacts={"result": "Stage 1 check passed.", "stage1_accuracy": result["score"]}
        )
    else:
        logger.info(f"Stdout: {result['stdout']}")
        logger.info(f"Stderr: {result['stderr']}")
        error_type = "UnknownError"
        if "Timeout" in result["error"]:
            error_type = "TimeoutError"
        elif "exited with code" in result["error"]:
            error_type = "ScriptCrash"
        elif "Could not parse" in result["error"]:
            error_type = "OutputParsingError"
        
        suggestion = "Check stderr for errors like ModuleNotFoundError, syntax errors, or runtime exceptions. Ensure the script prints 'mean_score: <value>'."
        if error_type == "TimeoutError":
            suggestion = "The script is too slow or stuck in a loop. Check for performance bottlenecks or infinite loops."
        
        return EvaluationResult(
            metrics={"combined_score": 0.0, "runs_successfully": 0.0, "error": error_type},
            artifacts={
                "error_type": error_type,
                "error_message": result['error'],
                "suggestion": suggestion,
                "stderr": result['stderr'][-1000:] if result['stderr'] else "N/A",
                "stdout": result['stdout'][-1000:] if result['stdout'] else "N/A"
            }
        )

def evaluate_stage2(program_path):
    print("--- Running Stage 2 Evaluation ---")
    return evaluate(program_path)