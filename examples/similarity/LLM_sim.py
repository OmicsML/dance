import csv
import json
import os
import sys

from openai import OpenAI
from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import PreservedScalarString

# ================= Configuration =================
# API Configuration
API_KEY = "sk-92265d8b2c044989b10ad59e3a27b56f"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen-plus"

# Paths and Server Configuration
SERVER_HOST = "211.87.232.112"
SERVER_PORT = 8000
BASE_PATH = "/mnt/nfs/zyxing/msu/dance_temp/dance/examples/search/generate_pseudocode/"
BASE_YAML_PATH = "/mnt/nfs/zyxing/msu/dance_temp/dance/examples/evolo/"
RESULT_CSV = "similarity_results_llm.csv"
SIMILARITY_MATRIX_CSV = "similarity_matrix.csv"

# Method List
ALL_METHODS = [
    "cta_scdeepsort", "cta_scheteronet", "domain_efnst", "domain_louvain", "domain_spagcn", "domain_stagate",
    "cta_scgat", "cta_scrgcl", "cta_graphcs", "domain_stlearn", 'domain_spagra'
]

# Initialize OpenAI Client
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# Import Knowledge Base (Mock if missing)
try:
    from GraphEvolve.lamarckian_knowledge_base import LamarckianKnowledgeBase
except ImportError:
    print("Warning: GraphEvolve module not found. Knowledge Base features will fail.")
    LamarckianKnowledgeBase = None

# ================= Core Functions =================


def load_method_code(method_name):
    """Reads the code content for a specific method."""
    file_path = os.path.join(BASE_PATH, f"{method_name}_openevolve_output/best/best_program.txt")
    if not os.path.exists(file_path):
        return None

    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()
        if '#Pseudocode-START' in content and '#Pseudocode-END' in content:
            content = content.split('#Pseudocode-START')[1].split('#Pseudocode-END')[0]
        return content.strip()
    except Exception as e:
        print(f"Error reading code for {method_name}: {e}")
        return None


def get_code_similarity_by_llm(code_a, code_b):
    """Uses LLM to compare logical similarity using Multi-Dimensional Weighted Scoring.

    Returns a float (0.0 - 1.0) and a reason string.

    """
    max_len = 20000  # Qwen-plus context window is large, but safe limit is good
    code_a_clip = code_a[:max_len]
    code_b_clip = code_b[:max_len]

    prompt = f"""
    You are a Senior Code Analysis Engine. Compare the following two Python code snippets (Code A and Code B).

    **Task:** Quantify the similarity between Code A and Code B across 4 specific dimensions.

    **Dimensions to Score (0-10 Scale):**
    1. **Algorithmic Logic (40% weight):** Do they use the same core algorithm family (e.g., both use GNNs, Clustering, or Matrix Factorization)?
       - 10: Identical algorithmic approach.
       - 5: Same problem domain, different mathematical approach.
       - 0: Completely different logic.
    2. **Structural Flow (30% weight):** Similarity in control flow (loops, conditionals), data flow pipelines, and modularity.
    3. **Goal/Semantics (20% weight):** Do they solve the exact same problem (e.g., both are for Spatial Transcriptomics)?
    4. **Surface Syntax (10% weight):** Similarity in variable naming style, specific library calls (e.g., scanpy, torch), and coding patterns.

    **CRITICAL INSTRUCTION:** - Be precise. Avoid "safe" middle scores (like 5) unless truly applicable.
    - Focus on finding *structural and algorithmic* similarities that would make one useful for optimizing the other.

    **Code A:**
    ```python
    {code_a_clip}
    ```

    **Code B:**
    ```python
    {code_b_clip}
    ```

    **Output Requirement:**
    Return ONLY a JSON object:
    {{
        "algo_score": <int 0-10>,
        "structure_score": <int 0-10>,
        "goal_score": <int 0-10>,
        "syntax_score": <int 0-10>,
        "reason": "<One sentence explaining the strongest similarity>"
    }}
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME, messages=[{
                "role":
                "system",
                "content":
                "You are a rigorous code similarity assessment system. Always output in valid JSON format."
            }, {
                "role": "user",
                "content": prompt
            }], response_format={"type": "json_object"}, temperature=0.1)

        result_text = response.choices[0].message.content
        data = json.loads(result_text)

        # --- Python-side Weighted Calculation ---
        # Weights: Algo(0.4) + Structure(0.3) + Goal(0.2) + Syntax(0.1)
        # This naturally produces granular floats like 0.74, 0.82, etc.
        weighted_sum = ((data.get('algo_score', 0) * 0.4) + (data.get('structure_score', 0) * 0.3) +
                        (data.get('goal_score', 0) * 0.2) + (data.get('syntax_score', 0) * 0.1))

        final_score = weighted_sum / 10.0  # Normalize to 0.0 - 1.0 range
        reason = data.get('reason', "No reason provided")

        return final_score, reason

    except Exception as e:
        print(f"LLM API Error: {e}")
        return 0.0, "LLM Error"


# ================= Matrix & Cache Logic =================


def get_pair_key(method_a, method_b):
    """Returns a standardized key for a pair of methods to ensure symmetry."""
    return tuple(sorted((method_a, method_b)))


def load_existing_matrix(csv_path):
    """Loads the similarity matrix CSV into a pairwise dictionary cache."""
    cache = {}
    if not os.path.exists(csv_path):
        print("ℹ️ No existing matrix file found. Starting fresh.")
        return cache

    try:
        with open(csv_path, encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            headers = next(reader)
            method_columns = headers[1:]

            for row in reader:
                row_method = row[0]
                for idx, score_str in enumerate(row[1:]):
                    col_method = method_columns[idx]
                    if score_str and score_str.strip() != "":
                        try:
                            score = float(score_str)
                            key = get_pair_key(row_method, col_method)
                            cache[key] = {'score': score, 'reason': "Loaded from Matrix CSV"}
                        except ValueError:
                            continue
        print(f"✅ Loaded matrix cache with {len(cache)} pairs.")
    except Exception as e:
        print(f"⚠️ Error loading matrix csv: {e}")

    return cache


def compute_full_matrix(methods, method_code_map, initial_cache):
    """Ensures every pair in methods has a score."""
    cache = initial_cache.copy()

    # ================= 改动开始 =================
    # 在这里填入你想强制重新计算的方法名（列表）
    # 只要配对中涉及这些方法，就会忽略 CSV 里的旧值，强制重跑 LLM
    FORCE_RERUN_METHODS = [
        "domain_stlearn",  # 举例：你想重跑这个
        "domain_spagra",  # 举例：还有这个
        "domain_stagate"
    ]
    # ================= 改动结束 =================

    total_pairs = (len(methods) * (len(methods) - 1)) // 2 + len(methods)
    print(f"\n🚀 Checking/Computing {total_pairs} pairs for {len(methods)} methods...")

    for i, method_a in enumerate(methods):
        for method_b in methods:
            # 1. Self comparison (如果是自己跟自己比，通常不需要重跑，直接设为 1.0)
            if method_a == method_b:
                key = get_pair_key(method_a, method_b)
                cache[key] = {'score': 1.0, 'reason': "Self comparison"}
                continue

            # 2. Check if pair exists
            key = get_pair_key(method_a, method_b)

            # ================= 改动逻辑 =================
            # 判断当前配对是否包含需要强制重跑的方法
            is_forced = (method_a in FORCE_RERUN_METHODS) or (method_b in FORCE_RERUN_METHODS)

            # 如果 key 在缓存里，且 不需要强制重跑，才跳过
            if key in cache and not is_forced:
                continue
            # ===========================================

            # 3. Compute via LLM
            print(f"   Computing: {method_a} vs {method_b} ...", end="", flush=True)

            code_a = method_code_map.get(method_a)
            code_b = method_code_map.get(method_b)

            if code_a and code_b:
                score, reason = get_code_similarity_by_llm(code_a, code_b)
                print(f" Score: {score:.4f}")
                cache[key] = {'score': score, 'reason': reason}
            else:
                print(" Missing Code!")
                cache[key] = {'score': 0.0, 'reason': "Code missing"}

    return cache


def save_matrix_csv(methods, cache, output_path):
    """Saves the cache as a symmetric matrix CSV."""
    try:
        with open(output_path, mode='w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow([''] + methods)

            for row_method in methods:
                row_data = [row_method]
                for col_method in methods:
                    key = get_pair_key(row_method, col_method)
                    entry = cache.get(key, {'score': 0.0})
                    row_data.append(f"{entry['score']:.4f}")  # Save with precision
                writer.writerow(row_data)
        print(f"✅ Matrix saved to {output_path}")
    except Exception as e:
        print(f"❌ Error saving matrix: {e}")


# ================= Update Config Logic =================


def update_method_config(target_method, best_candidate, score, principles_str):
    """Updates the YAML configuration for the target method."""
    # Note: Using your original logic to construct the file path
    # If target_method contains 'domain_' or 'cta_', ensure folder structure matches
    init_yaml_path = os.path.join(BASE_YAML_PATH, f"{target_method}/config.yaml")

    if not os.path.exists(init_yaml_path):
        # Fallback check: sometimes folder names don't exactly match method keys
        # But assuming your setup is consistent based on previous code.
        return f"Config file not found at {init_yaml_path}"

    try:
        ryaml = YAML()
        ryaml.default_flow_style = False
        ryaml.indent(mapping=2, sequence=4, offset=2)

        with open(init_yaml_path, encoding='utf-8') as f:
            config = ryaml.load(f)

        if 'prompt' in config and 'system_message' in config['prompt']:

            full_message = config['prompt']['system_message'] + \
               f"\n\n**{best_candidate} is identified as a structurally similar algorithm, sharing a similarity score of {score:.4f} " \
               f"with the algorithm currently under optimization. Consequently, the following graph construction rules may serve as a useful reference:**\n" \
               f"```{principles_str}```"

            config['prompt']['system_message'] = PreservedScalarString(full_message)

            output_yaml_path = init_yaml_path.replace('config.yaml', 'evolved_config.yaml')
            with open(output_yaml_path, 'w', encoding='utf-8') as f:
                ryaml.dump(config, f)

            return f"Success (Saved to {os.path.basename(output_yaml_path)})"
        else:
            return "YAML missing 'prompt' field"

    except Exception as e:
        return f"YAML Error: {e}"


# ================= Main =================


def main():
    # 1. Load Code
    print("--- 1. Loading Method Codes ---")
    method_code_map = {}
    for m in ALL_METHODS:
        code = load_method_code(m)
        if code:
            method_code_map[m] = code
    print(f"Loaded {len(method_code_map)} codes.")

    # 2. Load Existing Matrix Cache (for Resume/Symmetry)
    print("--- 2. Loading Existing Matrix ---")
    pairwise_cache = load_existing_matrix(SIMILARITY_MATRIX_CSV)

    # 3. Compute Missing Pairs (Fill the Matrix)
    print("--- 3. Computing Similarity Matrix ---")
    pairwise_cache = compute_full_matrix(ALL_METHODS, method_code_map, pairwise_cache)

    # 4. Save Complete Matrix
    save_matrix_csv(ALL_METHODS, pairwise_cache, SIMILARITY_MATRIX_CSV)

    # 5. Find Best Matches & Update YAMLs
    print("\n--- 4. Updating Configurations ---")

    summary_results = []

    # Initialize KB
    kb = None
    if LamarckianKnowledgeBase:
        try:
            # Assuming KB needs no args or args are correct
            kb = LamarckianKnowledgeBase(host=SERVER_HOST, port=SERVER_PORT)
        except Exception as e:
            print(f"⚠️ Could not connect to Knowledge Base: {e}")

    for target_method in ALL_METHODS:
        print(f"\nProcessing: {target_method}")

        # Find best match from the pairwise_cache
        best_candidate = None
        highest_score = -1.0
        best_reason = ""

        for candidate in ALL_METHODS:
            if candidate == target_method:
                continue

            key = get_pair_key(target_method, candidate)
            data = pairwise_cache.get(key, {'score': 0.0, 'reason': ''})

            if data['score'] > highest_score:
                highest_score = data['score']
                best_candidate = candidate
                best_reason = data.get('reason', '')

        print(f"  -> Best Match: {best_candidate} (Score: {highest_score:.4f})")

        status = "Skipped (No KB)"
        error_msg = ""

        # Retrieve Knowledge & Update
        if kb and best_candidate:
            try:
                # Assuming get_memory_by_task logic is correct
                principles, _ = kb.get_memory_by_task(best_candidate)

                # Handle cases where principles might be list or dict
                if isinstance(principles, list):
                    principles_content = [p.get('content', str(p)) for p in principles]
                    principles_str = "\n".join(principles_content)
                else:
                    principles_str = str(principles)

                status = update_method_config(target_method, best_candidate, highest_score, principles_str)
            except Exception as e:
                status = "Error"
                error_msg = str(e)
                print(f"  -> Error updating config: {e}")

        summary_results.append([target_method, best_candidate, f"{highest_score:.4f}", best_reason, status, error_msg])

    # 6. Save Summary CSV
    try:
        headers = ["Input Method", "Most Similar Method", "LLM Score", "Reason", "Status", "Error Message"]
        with open(RESULT_CSV, mode='w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(summary_results)
        print(f"\n✅ Summary saved to {RESULT_CSV}")
    except Exception as e:
        print(f"❌ Error saving summary: {e}")


if __name__ == "__main__":
    main()
