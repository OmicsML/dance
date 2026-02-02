import os
import json
from pathlib import Path

def load_best_program_info():
    """读取所在文件夹中以cta和domain开头的文件夹，然后以json的形式加载其中的best_program_info.json"""

    # 获取当前脚本所在目录
    current_dir = Path(__file__).parent

    # 存储结果的字典
    results = {}

    # 遍历当前目录下的所有文件夹
    for folder_name in os.listdir(current_dir):
        folder_path = current_dir / folder_name

        # 检查是否为文件夹且以cta或domain开头
        if folder_path.is_dir() and (folder_name.startswith('cta') or folder_name.startswith('domain')):
            json_file_path = folder_path / 'openevolve_output' / 'best' / 'best_program_info.json'

            # 检查json文件是否存在
            if json_file_path.exists():
                try:
                    with open(json_file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        results[folder_name] = data
                        print(f"成功加载: {folder_name}")
                except Exception as e:
                    print(f"加载 {folder_name} 时出错: {e}")
            else:
                print(f"文件不存在: {json_file_path}")

    return results

def extract_specific_scores(results):
    """从metrics中提取除去combined_score以外，除去后缀以inner_score结尾以外的所有score"""

    extracted_scores = {}

    for folder_name, data in results.items():
        if 'metrics' in data:
            metrics = data['metrics']
            folder_scores = {}

            # 遍历所有metrics中的键值对
            for key, value in metrics.items():
                # 条件：以_score结尾，但不是combined_score，也不是以inner_score结尾
                if (key.endswith('_score') and
                    key != 'combined_score' and
                    not key.endswith('_inner_score') and key!='reliability_score' and key != 'avg_speed_score'):
                    folder_scores[key] = value

            extracted_scores[folder_name] = folder_scores

    return extracted_scores

def calculate_combined_score(results):
    """计算组合得分：score均值 * 0.8 + 0.2 * avg_speed_score"""

    test_combined_scores = {}

    for folder_name, data in results.items():
        if 'metrics' in data:
            metrics = data['metrics']

            # 提取要计算均值的score（排除combined_score, inner_score, reliability_score, avg_speed_score）
            target_scores = []
            avg_speed_score = None

            for key, value in metrics.items():
                if (key.endswith('_score') and
                    key != 'combined_score' and
                    not key.endswith('_inner_score') and
                    key != 'reliability_score'):

                    if key == 'avg_speed_score':
                        avg_speed_score = value
                    else:
                        target_scores.append(value)

            # 计算均值
            if target_scores and avg_speed_score is not None:
                avg_score = sum(target_scores) / len(target_scores)
                test_combined_score = avg_score * 0.8 + avg_speed_score * 0.2
                test_combined_scores[folder_name] = {
                    'avg_score': avg_score,
                    'avg_speed_score': avg_speed_score,
                    'test_combined_score': test_combined_score,
                    'num_scores': len(target_scores)
                }

    return test_combined_scores

if __name__ == "__main__":
    # 加载所有best_program_info.json文件
    all_results = load_best_program_info()

    # 打印加载结果摘要
    print(f"\n总共加载了 {len(all_results)} 个文件夹的结果")

    # 提取特定的score
    specific_scores = extract_specific_scores(all_results)

    # 计算组合得分
    combined_scores = calculate_combined_score(all_results)

    # # 打印提取的score摘要
    # print(f"\n提取的score结果:")
    # for folder_name, scores in specific_scores.items():
    #     print(f"- {folder_name}: {len(scores)} 个score")
    #     for score_name, score_value in scores.items():
    #         print(f"  * {score_name}: {score_value:.6f}")

    # 将组合得分结果保存到文件
    with open('combined_scores_result.txt', 'w', encoding='utf-8') as f:
        f.write(f"组合得分结果 (均值*0.8 + 0.2*avg_speed_score):\n")
        for folder_name, score_info in combined_scores.items():
            f.write(f"- {folder_name}:\n")
            f.write(f"  * 均值: {score_info['avg_score']:.6f}\n")
            f.write(f"  * avg_speed_score: {score_info['avg_speed_score']:.6f}\n")
            f.write(f"  * 组合得分: {score_info['test_combined_score']:.6f}\n")
            f.write(f"  * score数量: {score_info['num_scores']}\n")
