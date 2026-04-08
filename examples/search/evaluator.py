"""Evaluator for the scDeepSort cell type annotation script."""
#TODO 改为deepeval
import json
import logging
import os
import re
import subprocess
import time
import traceback

import numpy as np
from openevolve.evaluation_result import EvaluationResult

from dance.modules.spatial.spatial_domain import spagcn
from dance.settings import EXAMPLESDIR

# Define the benchmarks based on the script's documentation.
# We use a reduced number of epochs (20) for faster evaluation compared to the
# original 300, while still being substantial enough to measure performance.


stage1_args = os.getenv("stage1_args")
BENCHMARKS_args = os.getenv("BENCHMARKS_args")
with open(f"{EXAMPLESDIR}/evolo/benchmarks_config.json") as f:
    BENCHMARKS = json.load(f)
stage1_args = BENCHMARKS[stage1_args]
BENCHMARKS = BENCHMARKS[BENCHMARKS_args]

# Timeout for each benchmark run in seconds.
# Training these models can take time, especially for data download on the first run.
BENCHMARK_TIMEOUT = 600000  # 1000 minutes
logger = logging.getLogger(__name__)


def _run_benchmark(program_path, benchmark_name, benchmark_args, timeout):
    """Helper function to run a single benchmark command."""
    # Use --cache to speed up subsequent runs by not reprocessing data.
    command = ["python", program_path, "--cache"] + benchmark_args

    start_time = time.time()
    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False  # Do not raise an exception on non-zero exit codes
        )

        end_time = time.time()
        execution_time = end_time - start_time

        # Check if the script crashed
        if process.returncode != 0:
            return {
                "status": "error",
                "score": 0.0,
                "time": execution_time,
                "error": f"Process exited with code {process.returncode}",
                "stdout": process.stdout,
                "stderr": process.stderr,
            }

        # Parse the output to find the mean score
        output = process.stdout
        match = re.search(r"mean_score:\s*([\d\.-]+)", output)
        inner_match = re.search(r"mean_inner_score:\s*([\d\.-]+)", output)
        if not match:
            return {
                "status": "error",
                "score": 0.0,
                "time": execution_time,
                "error": "Could not parse 'mean_score' from output.",
                "stdout": output,
                "stderr": process.stderr,
            }

        score = float(match.group(1))
        inner_score = float(inner_match.group(1))
        return {
            "status": "success",
            "score": score,
            "inner_score": inner_score,
            "time": execution_time,
            "error": None,
            "stdout": output,
            "stderr": process.stderr,
        }

    except subprocess.TimeoutExpired as e:
        return {
            "status": "error",
            "score": 0.0,
            "time": timeout,
            "error": f"Timeout of {timeout} seconds exceeded.",
            "stdout": e.stdout if e.stdout else "",
            "stderr": e.stderr if e.stderr else "",
        }
    except Exception as e:
        return {
            "status": "error",
            "score": 0.0,
            "time": time.time() - start_time,
            "error": f"An unexpected error occurred: {str(e)}",
            "stdout": "",
            "stderr": traceback.format_exc(),
        }


def evaluate(program_path):
    """Evaluates the script by running it against multiple benchmark datasets.

    The score is based on the average accuracy across the benchmarks. The script's
    ability to run without errors (reliability) is also factored in.

    Dependencies: The evaluation environment must have the following packages installed:
    - dance-cognition
    - torch
    - dgl
    - scikit-learn

    """

    results = {}
    successful_runs = []
    failed_runs = []

    for name, args in BENCHMARKS.items():
        print(f"--- Running benchmark: {name} ---")
        result = _run_benchmark(program_path, name, args, BENCHMARK_TIMEOUT)
        results[name] = result

        if result["status"] == "success":
            successful_runs.append(result)
            print(
                f"Success! Score: {result['score']:.4f}, Inner Score: {result['inner_score']:.4f}, Time: {result['time']:.2f}s"
            )
        else:
            failed_runs.append(result)
            print(f"Failed! Error: {result['error']}")
            logger.info(f"Stdout: {result['stdout']}")
            logger.info(f"Stderr: {result['stderr']}")

    if not successful_runs:
        error_details = {
            "error_type": "AllBenchmarksFailed",
            "error_message": "All benchmark datasets failed to run successfully.",
            "suggestion":
            "Check the detailed results for each benchmark. Common issues include timeouts, missing dependencies, or runtime errors in the code.",
            "failed_benchmarks": {
                name: res["error"]
                for name, res in results.items() if res["status"] == "error"
            },
            "sample_stderr": failed_runs[0]['stderr'][-1000:] if failed_runs and failed_runs[0]['stderr'] else "N/A"
        }
        return EvaluationResult(
            metrics={
                "combined_score": 0.0,
                "avg_accuracy": 0.0,
                "reliability_score": 0.0,
                "error": "All benchmarks failed."
            }, artifacts=error_details)

    # Calculate metrics based on successful runs
    avg_accuracy = np.mean([res["score"] for res in successful_runs])
    avg_inner_accuracy = np.mean([res["inner_score"] for res in successful_runs])
    avg_time = np.mean([res["time"] for res in successful_runs])
    reliability_score = len(successful_runs) / len(BENCHMARKS)

    # Speed score (higher is better). Normalized around 5 minutes (300s).
    speed_score = 1.0 / (1.0 + avg_time / 300.0)

    # Combined score prioritizes accuracy, with a bonus for reliability.
    combined_score = (0.7 * avg_inner_accuracy) + (0.2 * reliability_score) + (0.1 * speed_score)

    artifacts = {
        "benchmark_summary": {
            name: {
                "status": res["status"],
                "inner_score": round(res["inner_score"], 4) if "inner_score" in res else None,
                "time": round(res["time"], 2),
                "error": res["error"] if "error" in res else None
            }
            for name, res in results.items()
        },
        "average_execution_time":
        f"{avg_time:.2f} seconds",
        "performance_overview":
        f"Achieved an train average accuracy of {avg_inner_accuracy:.4f} across {len(successful_runs)} successful benchmarks."
    }
    metrics = {
        "combined_score": float(combined_score),
        "avg_accuracy": float(avg_accuracy),
        "avg_inner_accuracy": float(avg_inner_accuracy),
        "reliability_score": float(reliability_score),
        "speed_score": float(speed_score),
    }
    for name, res in results.items():
        metrics[f"{name}_score"] = float(res["score"]) if "score" in res else 0
        metrics[f"{name}_inner_score"] = float(res["inner_score"]) if "inner_score" in res else 0
        metrics[f"{name}_time"] = float(res["time"])
        metrics[f"{name}_status"] = res["status"]
    return EvaluationResult(metrics=metrics, artifacts=artifacts)


def evaluate_stage1(program_path):
    """A quick first-stage evaluation.

    It runs a single, simple benchmark with very few epochs to check if the script is
    syntactically correct and runs without crashing. This acts as a fast filter for
    invalid programs.

    """
    print("--- Running Stage 1 Evaluation ---")

    # Use the simplest benchmark with only 2 epochs for a quick check.

    timeout = 240  # 4 minutes timeout for the initial run (data download can be slow)

    result = _run_benchmark(program_path, "Stage1_Check", stage1_args, timeout)
    if result["status"] == "success":
        return EvaluationResult(metrics={
            "combined_score": 1.0,
            "runs_successfully": 1.0
        }, artifacts={
            "result": "Stage 1 check passed.",
            "stage1_accuracy": result["score"]
        })
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
            metrics={
                "combined_score": 0.0,
                "runs_successfully": 0.0,
                "error": error_type
            }, artifacts={
                "error_type": error_type,
                "error_message": result['error'],
                "suggestion": suggestion,
                "stderr": result['stderr'][-1000:] if result['stderr'] else "N/A",
                "stdout": result['stdout'][-1000:] if result['stdout'] else "N/A"
            })


def evaluate_stage2(program_path):
    """Second stage evaluation with more thorough testing across all benchmarks.

    This is the main evaluation function.

    """
    print("--- Running Stage 2 Evaluation ---")
    return evaluate(program_path)
