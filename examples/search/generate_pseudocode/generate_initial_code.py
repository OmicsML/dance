#函数1：读取算法->生成5个类似的算法->存储到csv中，传进hf里面
#函数2：读取算法->生成初始伪代码
import argparse
import os

import pandas as pd
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

base_path = "/mnt/nfs/zyxing/msu/dance_temp/dance/dance/modules/"
# "cta_scgat","cta_scrgcl",'cta_graphcs','domain_stlearn'
path_dict = {
    "cta_scdeepsort": "single_modality/cell_type_annotation/scdeepsort.py",
    "cta_scheteronet": "single_modality/cell_type_annotation/scheteronet.py",
    "domain_efnst": "spatial/spatial_domain/EfNST.py",
    "domain_louvain": "spatial/spatial_domain/louvain.py",
    "domain_spagcn": "spatial/spatial_domain/spagcn.py",
    "domain_stagate": "spatial/spatial_domain/stagate.py",
    "cta_scgat": "single_modality/cell_type_annotation/scgat.py",
    "cta_scrgcl": "single_modality/cell_type_annotation/scrgcl.py",
    "cta_graphcs": "single_modality/cell_type_annotation/graphcs.py",
    "domain_stlearn": "spatial/spatial_domain/stlearn.py",
    "domain_spagra": "spatial/spatial_domain/spaGRA.py"
}

from openai import OpenAI

# 初始化客户端（建议放在全局，避免每次调用函数都重新初始化）
api_key = "sk-92265d8b2c044989b10ad59e3a27b56f"
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
client = OpenAI(api_key=api_key, base_url=base_url)
model_name = "qwen-plus"

dataset_name = "zhongyuxing/Graph_Structure_Learning_Pseudocode_new_new"
split = "train"


def generate_code(file_path, method_name, num_samples=2):
    with open(file_path) as f:
        content = f.read()
    dataset = load_dataset(dataset_name, split="train")
    filtered_dataset = dataset.filter(lambda example: example['method'] != method_name)
    new_data = [{'method': method_name, 'code': content}]
    for i in range(num_samples):
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    'role':
                    'system',
                    'content': ("You are a senior algorithm engineer. Your task is to refactor the provided algorithm "
                                "to vary its implementation details. Specifically, you should rename variables, "
                                "remove redundant code, and rewrite comments. Crucially, ensure the execution logic "
                                "and output results remain exactly the same (functionally identical). "
                                "Please output only the code.")
                },
                {
                    'role': 'user',
                    # 这里直接放入代码字符串即可
                    'content': content
                }
            ])
        code = completion.choices[0].message.content
        new_data.append({'method': method_name, 'code': code})
        print(code == content)
    new_dataset = Dataset.from_list(new_data, features=filtered_dataset.features)

    # 4. 拼接 (Concatenate)
    # 把过滤后的老数据和新数据拼在一起
    final_dataset = concatenate_datasets([filtered_dataset, new_dataset])

    # 5. (可选) 推送回 Hub
    final_dataset.push_to_hub(dataset_name, split="train")


def generate_clrs_pseudocode(file_path, method_name):
    """输入源代码，返回 CLRS 风格的伪代码."""
    with open(file_path) as f:
        content = f.read()
    response = client.chat.completions.create(
        model=model_name,
        # 温度建议设为 0 ~ 0.2，保证它老老实实翻译，不要瞎发挥
        temperature=0.1,
        messages=[{
            'role':
            'system',
            'content':
            """
                        You are an expert computer scientist acting as a strict pseudocode generation engine.
                        Your task is to translate the provided source code into **Algol-like pseudocode** (resembling Pascal or Algol-60).

                        ### 1. Output Format Constraints (CRITICAL):
                        - **NO Conversational Text**: Do not say "Here is the code", "Sure", or any intro/outro.
                        - **NO Explanations**: Do not explain the logic.
                        - **Markdown Only**: Output **ONLY** the pseudocode inside a markdown code block.
                        - **Start Immediately**: The very first character must be the backtick (`).

                        ### 2. Algol-like Style Conventions:
                        - **Assignment**: Use the colonequals operator (:=) for assignments (e.g., x := x + 1).
                        - **Block Delimiters**: Explicitly use keywords to denote blocks, such as **begin** ... **end**, or specific closers like **end if**, **end for**, **end while**.
                        - **Control Structures**:
                        - **if** condition **then** ... **else** ... **end if**
                        - **while** condition **do** ... **end while**
                        - **for** var := start **to** end **do** ... **end for**
                        - **Functions**: Start with **procedure** or **function**.

                        ### 3. Abstraction & Simplification Rules:
                        - **Focus on Algorithm Flow**: Capture the high-level intent and flow of the algorithm. Omit low-level implementation details (e.g., memory management, library imports, boilerplate).
                        - **Summarize Complex Logic**: You **SHOULD** summarize complex or verbose operations into high-level descriptive actions.
                        - *Example*: Instead of writing out a QuickSort loop, write `Sort(array)`.
                        - *Example*: Instead of showing 10 lines of regex parsing, write `ParseInput(data)`.
                        - **Remove Noise**: Omit logging, debug prints, assertion checks, and complex error handling (try/catch blocks) unless they are central to the algorithm's logic.
                        - **Merge Steps**: Trivial variable initializations or temporary data shuffling can be merged or omitted to keep the pseudocode clean.
                        """
        }, {
            'role': 'user',
            'content': f"Source Code:\n{content}"
        }])
    with open(f"pseudocode_{method_name}_prompt.txt", "w") as f:
        output = content + "The following is pseudocode generated based on the above algorithm:\n\n\n" + "#Pseudocode-START\n" + response.choices[
            0].message.content + "\n\n\n#Pseudocode-END"
        f.write(output)


if __name__ == "__main__":
    # 定义完整列表
    ALL_METHODS = [
        "cta_scdeepsort", "cta_scheteronet", "domain_efnst", "domain_louvain", "domain_spagcn", "domain_stagate",
        "cta_scgat", "cta_scrgcl", "cta_graphcs", "domain_stlearn", "domain_spagra"
    ]

    parser = argparse.ArgumentParser(description="Evaluate prompts on GEPA benchmark datasets")
    parser.add_argument(
        "--method_name",
        type=str,
        # 允许输入 'all' 来运行所有，或者是列表中的具体某一个
        choices=ALL_METHODS + ["all"],
        help="Method to generate, or 'all' for batch processing",
        default="all"  # 默认改为跑全部，或者你可以设为 None
    )

    args = parser.parse_args()

    # 逻辑判断：如果是 'all'，则使用完整列表；否则使用单个列表
    target_methods = ALL_METHODS if args.method_name == "all" else [args.method_name]

    for method in target_methods:
        print(f"=== Processing: {method} ===")
        # 确保 path_dict 里有这个 key
        if method in path_dict:
            file_path = os.path.join(base_path, path_dict[method])
            generate_code(file_path, method)
            generate_clrs_pseudocode(file_path, method)
        else:
            print(f"Warning: {method} not found in path_dict.")
