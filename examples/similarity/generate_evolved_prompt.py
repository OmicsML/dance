#1 输入指定的方法，根据欧几里得距离找到最相似的npy，然后根据npy的名称找到它的best里面的图预处理方法



#2 将这个图预处理方法返回到evo
import os

from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import PreservedScalarString

yaml = YAML()
yaml.indent(mapping=2, sequence=2, offset=2)
yaml.width = 4096 

def find_most_similar_method(input_method_name, data_dir="data"):
    """
    输入指定的方法，根据欧几里得距离找到最相似的npy文件，
    然后根据npy的名称找到它的best里面的图预处理方法

    Args:
        input_method_name: 输入的方法名称
        data_dir: 存储embedding向量的目录

    Returns:
        dict: 包含最相似方法的信息和图预处理方法
    """
    # 1. 获取输入方法的embedding向量
    base_path = "/mnt/nfs/zyxing/msu/dance_temp/dance/examples/evolo/"
    # 获取输入方法的embedding向量

    input_vector = np.load(f"data/{input_method_name}_embedding.npy")
    input_task=input_method_name.split('_')[0]
    # 2. 加载所有现有的embedding向量
    
    # embedding_files = [f for f in os.listdir(data_dir) if f.endswith('_embedding.npy')]
    embedding_files=[]
    for f in os.listdir(data_dir):
        if f.endswith('_embedding.npy'):
            task=f.split('_')[0]
            if input_task==task:
                embedding_files.append(f)
    
    
    if not embedding_files:
        raise FileNotFoundError(f"在{data_dir}目录下没有找到embedding文件")

    similarities = {}

    for emb_file in embedding_files:
        # 从文件名提取方法名称
        method_name = emb_file.replace('_embedding.npy', '')

        # 跳过输入方法本身
        if method_name == input_method_name:
            continue

        # 加载embedding向量
        emb_path = os.path.join(data_dir, emb_file)
        try:
            existing_vector = np.load(emb_path)

            # 计算余弦距离（相似度，距离越小越相似）
            distance = cosine(input_vector, existing_vector)
            similarities[method_name] = distance

        except Exception as e:
            print(f"加载{emb_file}时出错: {e}")
            continue

    if not similarities:
        raise ValueError("没有找到其他方法的embedding向量进行比较")

    # 3. 找到距离最小（最相似）的方法
    most_similar_method = min(similarities, key=similarities.get)
    min_distance = similarities[most_similar_method]

    print(f"输入方法: {input_method_name}")
    print(f"最相似的方法: {most_similar_method} (距离: {min_distance:.4f})")

    # 4. 读取最相似方法的图预处理方法
    best_path = os.path.join(base_path, f"{most_similar_method}/openevolve_output/best/best_program.py")
    best_yaml_path = os.path.join(base_path, f"{most_similar_method}/config.yaml")

    if not os.path.exists(best_path):
        raise FileNotFoundError(f"找不到最相似方法的best文件: {best_path}")

    with open(best_path,"r") as f:
        code=f.read()
        code=code.split('# EVOLVE-BLOCK-START')[1].split("# EVOLVE-BLOCK-END")[0]

    # 读取best_yaml_path并将code添加到其中的system message部分
    with open(best_yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f)

    # 将代码添加到system message的末尾
    if 'prompt' in config and 'system_message' in config['prompt']:
        full_message = config['prompt']['system_message']+f"\n\n**Here is the preprocessing method from {most_similar_method}. It is similar to the current method and serves as a reference for the implementation:**\n```python\n{code.strip()}\n```"
        config['prompt']['system_message'] = PreservedScalarString(full_message)
    output_yaml_path=best_yaml_path.replace('config.yaml','evolved_config.yaml')

    with open(output_yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f)


import argparse
import numpy as np
from scipy.spatial.distance import cosine


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate prompts on GEPA benchmark datasets")
    parser.add_argument(
        "--method_name",
        type=str,
        choices=["cta_scdeepsort", "cta_scheteronet", "domain_efnst", "domain_louvain","domain_spagcn","domain_stagate"],
        help="Method to generate",
        default="cta_scheteronet"
    )

    args = parser.parse_args()
    find_most_similar_method(args.method_name)