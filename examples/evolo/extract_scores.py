import os
import re
import csv
from pathlib import Path


def parse_combined_scores_file(file_path):
    """从combined_scores_result文件中提取数据"""
    
    # 从文件名解析is_evoloved和alpha
    filename = os.path.basename(file_path)
    
    # 默认值
    is_evoloved = False
    alpha = 0.8  # 默认alpha值（从summary_result.py看，当alpha!=0.8时才会在文件名中显示）
    
    # 检查是否包含evoloved
    if '_evoloved' in filename:
        is_evoloved = True
    
    # 检查是否包含alpha值
    alpha_match = re.search(r'alpha([\d.]+)', filename)
    if alpha_match:
        alpha_str = alpha_match.group(1).rstrip('.')
        alpha = float(alpha_str)
    
    # 解析文件内容
    results = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
        current_folder = None
        for line in lines:
            line = line.strip()
            
            # 匹配文件夹名行（以-开头的行）
            folder_match = re.match(r'^- (.+):$', line)
            if folder_match:
                current_folder = folder_match.group(1)
                results[current_folder] = {
                    'combined_score': None,
                    'initial_avg_speed_score': None,
                    'avg_speed_score': None,
                    'test_combined_score_std': None,
                    'test_combined_score': None,
                    'num_scores': None,
                    'mean_target_scores': None
                }
                continue
            
            # 匹配属性行
            if current_folder and line.startswith('*'):
                # 移除开头的*和空格
                line = line[1:].strip()
                
                # 匹配各种属性
                patterns = {
                    'combined_score': r'均值: ([\d.]+)',
                    'initial_avg_speed_score': r'初始avg_speed_score: ([\d.]+)',
                    'avg_speed_score': r'avg_speed_score: ([\d.]+)',
                    'test_combined_score_std': r'组合得分标准差: ([\d.]+)',
                    'test_combined_score': r'组合得分: ([\d.]+)',
                    'num_scores': r'score数量: (\d+)',
                    'mean_target_scores': r'均值target_scores: ([\d.]+)'
                }
                
                for key, pattern in patterns.items():
                    match = re.search(pattern, line)
                    if match:
                        if key == 'num_scores':
                            results[current_folder][key] = int(match.group(1))
                        else:
                            results[current_folder][key] = float(match.group(1))
                        break
    
    return is_evoloved, alpha, results


def process_all_files(input_dir, output_csv):
    """处理所有combined_scores_result文件并生成CSV"""
    
    input_path = Path(input_dir)
    
    # 收集所有文件
    all_data = []
    
    for file_path in input_path.glob('combined_scores_result*.txt'):
        # 排除自身（如果已经存在）
        if 'extract_scores' in str(file_path):
            continue
            
        print(f"处理文件: {file_path.name}")
        
        try:
            is_evoloved, alpha, results = parse_combined_scores_file(file_path)
            
            for folder_name, scores in results.items():
                row = {
                    'file_source': file_path.name,
                    'is_evoloved': is_evoloved,
                    'alpha': alpha,
                    'folder_name': folder_name
                }
                row.update(scores)
                all_data.append(row)
                
        except Exception as e:
            print(f"处理文件 {file_path.name} 时出错: {e}")
    
    # 获取所有可能的score列名
    score_columns = ['combined_score', 'initial_avg_speed_score', 'avg_speed_score', 
                     'test_combined_score_std', 'test_combined_score', 'num_scores', 'mean_target_scores']
    
    # 写入CSV
    fieldnames = ['file_source', 'is_evoloved', 'alpha', 'folder_name'] + score_columns
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        # 按is_evoloved、alpha、folder_name排序
        all_data.sort(key=lambda x: (not x['is_evoloved'], x['alpha'], x['folder_name']))
        
        for row in all_data:
            writer.writerow(row)
    
    print(f"\n成功写入 {output_csv}")
    print(f"总共处理了 {len(all_data)} 条记录")
    
    return all_data


def create_separate_csvs(all_data, output_dir):
    """为每个原始文件创建单独的CSV"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 按源文件分组
    file_groups = {}
    for row in all_data:
        source_file = row['file_source']
        if source_file not in file_groups:
            file_groups[source_file] = []
        file_groups[source_file].append(row)
    
    score_columns = ['combined_score', 'initial_avg_speed_score', 'avg_speed_score', 
                     'test_combined_score_std', 'test_combined_score', 'num_scores', 'mean_target_scores']
    
    fieldnames = ['file_source', 'is_evoloved', 'alpha', 'folder_name'] + score_columns
    
    for source_file, rows in file_groups.items():
        # 生成输出文件名
        base_name = source_file.replace('.txt', '_extracted.csv')
        output_file = output_path / base_name
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            rows.sort(key=lambda x: x['folder_name'])
            writer.writerows(rows)
        
        print(f"创建: {output_file}")


if __name__ == "__main__":
    # 设置路径
    input_directory = "/mnt/nfs/zyxing/msu/dance_temp/dance/examples/evolo"
    combined_csv = os.path.join(input_directory, "all_combined_scores.csv")
    
    # 处理所有文件，生成汇总CSV
    all_data = process_all_files(input_directory, combined_csv)
    
    # 为每个原始文件创建单独的CSV
    separate_csv_dir = os.path.join(input_directory, "extracted_scores")
    create_separate_csvs(all_data, separate_csv_dir)
    
    print("\n处理完成!")
