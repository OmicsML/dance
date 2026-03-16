"""
从 graph_automl 的结果中提取与 origin.yaml 默认参数匹配的 weighted_score_mean
"""
import csv
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def load_origin_config(config_path: str) -> Dict:
    """加载 origin.yaml 配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        # 修复 YAML 格式问题（domain_spaGRA 缺少冒号）
        content = f.read()
        # 修复: "domain_spaGRA" 应该是 "domain_spaGRA:"
        content = content.replace('\ndomain_spaGRA\n', '\ndomain_spaGRA:\n')
        config = yaml.safe_load(content)
    return config


def parse_csv_params(csv_header: List[str], row: List[str]) -> Dict[str, str]:
    """
    解析 CSV 行的参数
    例如: 
    - header: ['params.2.SpaGCNGraph.alpha', 'params.2.SpaGCNGraph.beta', ...]
    - row: ['2.5', '49', ...]
    返回: {'alpha': '2.5', 'beta': '49'}
    """
    params = {}
    # CSV 的前几列是参数列
    for i, col_name in enumerate(csv_header):
        if col_name.startswith('params.'):
            # 提取参数名：取最后一部分
            param_name = col_name.split('.')[-1]
            if i < len(row):
                params[param_name] = str(row[i])
    return params


def normalize_value(value) -> str:
    """标准化参数值用于比较"""
    if isinstance(value, (int, float)):
        return str(value)
    return str(value).strip()


def find_matching_row(csv_path: str, default_params: Dict) -> Optional[Tuple[Dict, float]]:
    """
    在 CSV 中找到与默认参数匹配的行
    
    Args:
        csv_path: CSV 文件路径
        default_params: 默认参数字典
    
    Returns:
        (匹配的行参数字典, weighted_score_mean) 或 None
    """
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        
        # 提取参数名列（以 params. 开头的列）
        param_cols = {}
        for i, col_name in enumerate(header):
            if col_name.startswith('params.'):
                param_name = col_name.split('.')[-1]
                param_cols[param_name] = i
        
        # 查找 weighted_score_mean 的列索引
        try:
            score_col_idx = header.index('weighted_score_mean')
        except ValueError:
            return None
        
        for row in reader:
            if len(row) <= max(param_cols.values()) or len(row) <= score_col_idx:
                continue
            
            # 提取行的参数
            row_params = {}
            for param_name, col_idx in param_cols.items():
                if col_idx < len(row):
                    row_params[param_name] = normalize_value(row[col_idx])
            
            # 检查是否匹配所有默认参数
            match = True
            for key, value in default_params.items():
                row_val = row_params.get(key)
                if row_val is None:
                    match = False
                    break
                # 处理浮点数精度问题
                try:
                    row_val_float = float(row_val)
                    default_float = float(value)
                    if abs(row_val_float - default_float) > 1e-6:
                        match = False
                        break
                except ValueError:
                    if row_val != str(value):
                        match = False
                        break
            
            if match:
                try:
                    score = float(row[score_col_idx])
                    return row_params, score
                except (ValueError, IndexError):
                    continue
    
    return None


def extract_default_params(folder_name: str, origin_config: Dict) -> Dict:
    """
    从 origin.yaml 配置中提取方法的默认参数
    
    Args:
        folder_name: 方法文件夹名（如 domain_spaGRA）
        origin_config: origin.yaml 配置
    
    Returns:
        默认参数字典
    """
    # 映射文件夹名到配置中的键名（支持大小写变化）
    config_key = folder_name
    
    # 尝试不同的键名形式
    possible_keys = [folder_name, folder_name.lower(), folder_name.replace('spagra', 'spaGRA')]
    
    method_config = None
    for key in possible_keys:
        if key in origin_config:
            method_config = origin_config[key]
            break
    
    if method_config is None:
        return {}
    
    if not isinstance(method_config, dict):
        return {}
    
    # 参数名映射（origin.yaml 中的参数名 -> CSV 中的参数名）
    param_mapping = {
        'rad_cutoff': 'rad_cutoff',
        'radius': 'radius',
        'n_neighbors': 'n_neighbors',
        'n_pcs': 'n_pcs',
        'crop_size': 'crop_size',
        'alpha': 'alpha',
        'beta': 'beta',
        'edge_ratio': 'edge_ratio',
        'distType': 'distType',
        'knn_num': 'knn_num',
        # distance_metrics 而不是 distance_metric
        'distance_metric': 'distance_metrics',
        'thres': 'thres',
        'normalize_edges': 'normalize_edges',
    }
    
    default_params = {}
    for key, value in method_config.items():
        mapped_key = param_mapping.get(key, key)
        default_params[mapped_key] = value
    
    return default_params


def find_result_csvs(base_dir: str, folder_name: str) -> List[Tuple[str, str]]:
    """
    查找方法的结果 CSV 文件
    
    Returns:
        [(alpha值, 文件路径), ...]
    """
    base_path = Path(base_dir)
    folder_path = base_path / folder_name
    
    if not folder_path.exists():
        return []
    
    results = []
    
    # 查找 summary_results_alpha*.csv
    for csv_file in folder_path.glob('summary_results_alpha*.csv'):
        # 从文件名提取 alpha 值
        alpha = csv_file.stem.replace('summary_results_alpha', '')
        results.append((alpha, str(csv_file)))
    
    # 如果没有带 alpha 的文件，查找 summary_results.csv（对应 alpha=0.8）
    summary_file = folder_path / 'summary_results.csv'
    if summary_file.exists():
        # 检查是否已经有 alpha=0.8 的结果
        has_08 = any(a == '0.8' for a, _ in results)
        if not has_08:
            results.append(('0.8', str(summary_file)))
    
    return sorted(results, key=lambda x: float(x[0]) if x[0] != 'N/A' else -1, reverse=True)


def generate_summary_table(base_dir: str, config_path: str, output_path: str):
    """
    生成综合结果表格
    
    Args:
        base_dir: graph_automl 目录
        config_path: origin.yaml 路径
        output_path: 输出文件路径
    """
    # 加载配置
    origin_config = load_origin_config(config_path)
    
    # 获取所有方法文件夹
    base_path = Path(base_dir)
    method_folders = [f.name for f in base_path.iterdir() if f.is_dir()]
    
    # 方法名映射（支持大小写变化）
    method_name_mapping = {
        "cta_scdeepsort": "ScDeepSort",
        "cta_scheteronet": "ScheteroNet",
        "cta_scgat": "ScGAT",
        "cta_scrgcl": "SCRGCL",
        "cta_graphcs": "GraphCS",
        "domain_efnst": "EfNST",
        "domain_louvain": "Louvain",
        "domain_spagcn": "SpaGCN",
        "domain_stagate": "STAGATE",
        "domain_stlearn": "stLearn",
        # 支持 spaGRA 大小写变化
        "domain_spagra": "SpaGRA",
        "domain_spaGRA": "SpaGRA",
    }
    
    # 收集结果
    results = []  # [(method, alpha, score, matched_params), ...]
    
    print("=" * 70)
    print("处理结果:")
    print("=" * 70)
    
    for folder_name in method_folders:
        method_display = method_name_mapping.get(folder_name, folder_name)
        
        # 获取默认参数
        default_params = extract_default_params(folder_name, origin_config)
        
        if not default_params:
            print(f"\n[{method_display}] 未找到默认参数配置")
            continue
        
        # 查找结果 CSV
        csv_files = find_result_csvs(base_dir, folder_name)
        
        if not csv_files:
            print(f"\n[{method_display}] 未找到结果文件")
            continue
        
        for alpha, csv_path in csv_files:
            match_result = find_matching_row(csv_path, default_params)
            
            if match_result:
                matched_params, score = match_result
                results.append({
                    'method': method_display,
                    'alpha': alpha,
                    'score': score,
                    'matched_params': matched_params
                })
                print(f"[{method_display}] α={alpha}: {score:.4f} (匹配参数: {matched_params})")
            else:
                print(f"[{method_display}] α={alpha}: 未找到匹配行 (默认参数: {default_params})")
    
    # 生成表格
    print("\n" + "=" * 70)
    print("综合结果表格:")
    print("=" * 70)
    
    # 获取所有唯一的 alpha 值（排除 N/A，按降序排列）
    alphas = sorted(
        set(r['alpha'] for r in results if r['alpha'] != 'N/A'),
        key=lambda x: float(x),
        reverse=True
    )
    
    # 按方法顺序排列
    method_order = [
        "ScDeepSort", "ScheteroNet", "EfNST", "Louvain", "SpaGCN",
        "STAGATE", "ScGAT", "SCRGCL", "GraphCS", "stLearn", "SpaGRA"
    ]
    
    # 创建结果字典
    scores = {}
    for r in results:
        method = r['method']
        alpha = r['alpha']
        scores[(method, alpha)] = r['score']
    
    # 打印表格
    header = ["Method"] + [f"α={a}" for a in alphas]
    print(f"{'Method':<15}", end="")
    for a in alphas:
        print(f"{'α='+a:>10}", end="")
    print()
    print("-" * (15 + 10 * len(alphas)))
    
    for method in method_order:
        print(f"{method:<15}", end="")
        for a in alphas:
            score = scores.get((method, a))
            if score is not None:
                print(f"{score:>10.4f}", end="")
            else:
                print(f"{'-':>10}", end="")
        print()
    
    # 保存到 CSV
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for method in method_order:
            row = [method]
            for a in alphas:
                score = scores.get((method, a))
                row.append(f"{score:.4f}" if score is not None else "-")
            writer.writerow(row)
    
    print(f"\n结果已保存到: {output_path}")
    
    return results


def load_default_scores(default_csv_path: str) -> Dict[str, Dict[str, str]]:
    """
    从 graph_automl_default_params_results.csv 加载默认分数

    Returns:
        {(method, alpha): score_str, ...}
    """
    scores = {}

    # 方法名映射（results_alpha*.csv 中的名称 -> graph_automl 中的名称）
    name_mapping = {
        "scHeteroNet": "ScheteroNet",
        "scGAT": "ScGAT",
        "scRGCL": "SCRGCL",
    }

    with open(default_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)

        # 解析 header: ["Method", "α=0.9", "α=0.8", ...]
        alphas = [h.replace('α=', '') for h in header[1:]]

        for row in reader:
            if not row:
                continue

            original_method = row[0]
            for i, alpha in enumerate(alphas):
                if i + 1 < len(row) and row[i + 1].strip():
                    try:
                        score_float = float(row[i + 1])
                        # 保存原始名称和映射名称
                        scores[(original_method, alpha)] = f"{score_float:.3f}"
                        # 如果有映射，也保存映射后的键
                        mapped_method = name_mapping.get(original_method)
                        if mapped_method:
                            scores[(mapped_method, alpha)] = f"{score_float:.3f}"
                    except ValueError:
                        pass

    return scores


def add_original_default_column(results_dir: str, default_csv_path: str):
    """
    为每个 results_alpha*.csv 添加 Original Default 列

    Args:
        results_dir: results_alpha*.csv 文件所在目录
        default_csv_path: graph_automl_default_params_results.csv 路径
    """
    results_path = Path(results_dir)
    default_scores = load_default_scores(default_csv_path)

    # 方法名映射（results_alpha*.csv 中的名称 -> graph_automl 中的名称）
    name_mapping = {
        "scHeteroNet": "ScheteroNet",
        "scGAT": "ScGAT",
        "scRGCL": "SCRGCL",
    }

    # 查找所有 results_alpha*.csv 文件
    results_files = sorted(results_path.glob('results_alpha*.csv'))

    print("=" * 70)
    print("为 results_alpha*.csv 添加 Original Default 列:")
    print("=" * 70)

    for csv_file in results_files:
        # 从文件名提取 alpha 值
        alpha = csv_file.stem.replace('results_alpha', '')

        # 读取原始文件
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)

        if not rows:
            continue

        header = rows[0]

        # 检查是否已经有 Original Default 列（精确匹配）
        header_lower = [h.strip() for h in header]
        has_default_col = 'Original Default' in header_lower

        # 如果已经有该列，更新缺失的值；否则添加新列
        if has_default_col:
            print(f"  {csv_file.name}: 已有 Original Default 列，更新缺失的值 (α={alpha})")
        else:
            # 添加列名
            header.append('Original Default')
            print(f"  {csv_file.name}: 添加 Original Default 列 (α={alpha})")

        # 为每一行添加/更新默认值
        default_col_idx = header.index('Original Default')
        updated_count = 0
        for i, row in enumerate(rows[1:], start=1):
            if not row:
                continue

            method = row[0]
            score = default_scores.get((method, alpha))

            # 如果找不到，尝试映射名称
            if not score and method in name_mapping:
                score = default_scores.get((name_mapping[method], alpha))

            if score:
                # 确保行有足够的列
                while len(row) <= default_col_idx:
                    row.append('')
                row[default_col_idx] = score
                updated_count += 1

        # 保存更新后的文件
        with open(csv_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)

        print(f"  {csv_file.name}: 添加了 {updated_count} 个 Original Default 值 (α={alpha})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="提取与默认参数匹配的 weighted_score_mean")
    parser.add_argument('--base-dir', type=str,
                       default='graph_automl',
                       help='graph_automl 目录')
    parser.add_argument('--config', type=str,
                       default='graph_automl/origin.yaml',
                       help='origin.yaml 配置文件路径')
    parser.add_argument('--output', type=str,
                       default='graph_automl_default_params_results.csv',
                       help='输出文件路径')
    parser.add_argument('--results-dir', type=str,
                       default=None,
                       help='results_alpha*.csv 文件所在目录（用于添加 Original Default 列）')
    parser.add_argument('--add-default', action='store_true',
                       help='为 results_alpha*.csv 添加 Original Default 列')

    args = parser.parse_args()

    # 使用脚本所在目录
    script_dir = Path(__file__).parent.resolve()
    base_dir = script_dir / args.base_dir
    config_path = script_dir / args.config
    output_path = script_dir / args.output

    if not base_dir.exists():
        print(f"错误: 目录 {base_dir} 不存在")
        exit(1)

    if not config_path.exists():
        print(f"错误: 配置文件 {config_path} 不存在")
        exit(1)

    # 生成综合结果表格
    generate_summary_table(str(base_dir), str(config_path), str(output_path))

    # 添加 Original Default 列（如果需要）
    if args.add_default or args.results_dir:
        results_dir = args.results_dir if args.results_dir else script_dir
        add_original_default_column(str(results_dir), str(output_path))
