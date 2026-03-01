import os
import sys
import csv
import numpy as np
from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import PreservedScalarString
# 假设这是您的自定义库，修正了类名拼写
from GraphEvolve.lamarckian_knowledge_base import LamarckianKnowledgeBase

# 初始化 YAML
yaml = YAML()
yaml.indent(mapping=2, sequence=2, offset=2)
yaml.width = 4096

# 全局配置
SERVER_HOST = "211.87.232.112"
SERVER_PORT = 8000
DATA_DIR = "data"  # embedding 存放目录
RESULT_CSV = "similarity_results.csv" # 结果输出文件

ALL_METHODS = [
    "cta_scdeepsort", "cta_scheteronet", "domain_efnst", "domain_louvain",
    "domain_spagcn", "domain_stagate", "cta_scgat", "cta_scrgcl",
    "cta_graphcs", "domain_stlearn"
]



def process_single_method(input_method_name, data_dir):
    """
    处理单个方法的寻找相似、获取知识、修改YAML流程
    返回: (dict) 用于写入CSV的结果信息
    """
    result_info = {
        "input_method": input_method_name,
        "most_similar_method": "N/A",
        "distance": "N/A",
        "status": "Failed",
        "error_msg": ""
    }

    print(f"\n{'='*10} Processing: {input_method_name} {'='*10}")

    # 1. 获取输入方法的embedding向量
    input_emb_path = os.path.join(data_dir, f"{input_method_name}_embedding.npy")
    if not os.path.exists(input_emb_path):
        msg = f"未找到输入方法的embedding文件: {input_emb_path}"
        print(msg)
        result_info["error_msg"] = msg
        return result_info

    try:
        input_vector = np.load(input_emb_path)
    except Exception as e:
        msg = f"加载输入向量失败: {e}"
        print(msg)
        result_info["error_msg"] = msg
        return result_info

    input_task = input_method_name.split('_')[0]

    # 2. 筛选同类型的embedding文件
    embedding_files = []
    if not os.path.exists(data_dir):
        msg = f"Data目录不存在: {data_dir}"
        result_info["error_msg"] = msg
        return result_info

    for f in os.listdir(data_dir):
        if f.endswith('_embedding.npy'):
            # 提取task前缀进行匹配
            task = f.split('_')[0]
            if input_task == task:
                embedding_files.append(f)
    
    if not embedding_files:
        msg = f"在{data_dir}目录下没有找到同类型(task={input_task})的embedding文件"
        print(msg)
        result_info["error_msg"] = msg
        return result_info

    # 3. 计算距离 (欧几里得距离)
    similarities = {}
    for emb_file in embedding_files:
        method_name = emb_file.replace('_embedding.npy', '')
        
        if method_name == input_method_name:
            continue

        emb_path = os.path.join(data_dir, emb_file)
        try:
            existing_vector = np.load(emb_path)
            
            # 使用欧几里得距离 (np.linalg.norm)
            # 如果需要余弦距离，请替换为 scipy.spatial.distance.cosine
            dist = np.linalg.norm(input_vector - existing_vector)
            similarities[method_name] = dist
            
        except Exception as e:
            print(f"加载对比向量 {emb_file} 时出错: {e}")
            continue

    if not similarities:
        msg = "没有找到其他有效的方法向量进行比较"
        print(msg)
        result_info["error_msg"] = msg
        return result_info

    # 4. 找到距离最小的方法
    most_similar_method = min(similarities, key=similarities.get)
    min_distance = similarities[most_similar_method]
    
    print(f"-> 最相似的方法: {most_similar_method} (欧氏距离: {min_distance:.4f})")
    
    result_info["most_similar_method"] = most_similar_method
    result_info["distance"] = round(min_distance, 4)

    # 5. 连接知识库获取 Principles
    try:
        kb = LamarckianKnowledgeBase(host=SERVER_HOST, port=SERVER_PORT)
        # print("✅ 知识库连接成功")
    except Exception as e:
        msg = f"❌ 知识库连接失败: {e}"
        print(msg)
        result_info["error_msg"] = msg
        return result_info

    try:
        principles = kb.get_principles(most_similar_method)
        if isinstance(principles, list):
            principles_str = "\n".join(principles)
        else:
            principles_str = str(principles)
    except Exception as e:
        msg = f"获取Principles失败: {e}"
        print(msg)
        result_info["error_msg"] = msg
        return result_info

    # 6. 修改 YAML 配置文件
    init_yaml_path = os.path.join(base_path, f"{input_method_name}/config.yaml")
    
    if not os.path.exists(init_yaml_path):
        msg = f"找不到配置文件: {init_yaml_path}"
        print(msg)
        result_info["error_msg"] = msg
        return result_info

    try:
        with open(init_yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.load(f)

        if 'prompt' in config and 'system_message' in config['prompt']:
            full_message = config['prompt']['system_message'] + \
                           f"\n\n**Here is the preprocessing method from {most_similar_method}. It is similar to the current method and has the following rules that can be learned from it:**\n```{principles_str}```"
            
            config['prompt']['system_message'] = PreservedScalarString(full_message)
            
            output_yaml_path = init_yaml_path.replace('config.yaml', 'evolved_config.yaml')
            with open(output_yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f)
            
            print(f"-> 已保存更新后的配置到: {output_yaml_path}")
            result_info["status"] = "Success"
        else:
            msg = "YAML中缺少 prompt 或 system_message 字段"
            print(msg)
            result_info["error_msg"] = msg
            
    except Exception as e:
        msg = f"读写YAML文件失败: {e}"
        print(msg)
        result_info["error_msg"] = msg

    return result_info

def main():
    # 准备 CSV 文件头
    headers = ["Input Method", "Most Similar Method", "Euclidean Distance", "Status", "Error Message"]
    
    # 收集结果
    results = []

    print(f"开始批量处理 {len(ALL_METHODS)} 个方法...")
    
    for method in ALL_METHODS:
        info = process_single_method(method, DATA_DIR)
        results.append(info)

    # 写入 CSV
    try:
        with open(RESULT_CSV, mode='w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for res in results:
                writer.writerow([
                    res["input_method"],
                    res["most_similar_method"],
                    res["distance"],
                    res["status"],
                    res["error_msg"]
                ])
        print(f"\n✅ 所有任务完成。统计结果已保存至: {os.path.abspath(RESULT_CSV)}")
    except Exception as e:
        print(f"\n❌ 写入CSV失败: {e}")

if __name__ == "__main__":
    main()