"""
将 combined_all_results.csv 转换为实验结果表格
1. 按 alpha 值分组
2. folder_name 重命名为 Method（需映射）
3. method 转为表格列名（需映射）
"""
import csv
from pathlib import Path


def map_folder_name(name: str) -> str:
    """将 folder_name 映射为更友好的名称"""
    mapping = {
        # 按用户指定顺序
        "domain_stagate": "STAGATE",
        "domain_spagcn": "SpaGCN",
        "domain_louvain": "Louvain",
        "cta_scdeepsort": "ScDeepSort",
        "cta_scheteronet": "scHeteroNet",
        "cta_scgat": "scGAT",
        "domain_spagra": "SpaGRA",
        "domain_efnst": "EfNST",
        "cta_scrgcl": "scRGCL",
        "cta_graphcs": "GraphCS",
        "domain_stlearn": "stLearn",
    }
    return mapping.get(name, name)


# Method 排列顺序（按用户指定）
METHOD_ORDER = [
    "domain_stagate",  # STAGATE
    "domain_spagcn",   # SpaGCN
    "domain_louvain",  # Louvain
    "cta_scdeepsort",  # ScDeepSort
    "cta_scheteronet", # scHeteroNet
    "cta_scgat",       # scGAT
    "domain_spagra",   # SpaGRA
    "domain_efnst",   # EfNST
    "cta_scrgcl",      # scRGCL
    "cta_graphcs",     # GraphCS
    "domain_stlearn",  # stLearn
]


def format_score(score: float) -> str:
    """格式化分数为三位小数"""
    if score is None:
        return "-"
    return f"{score:.3f}"


def map_method(method: str) -> str:
    """将 method 映射为列名"""
    mapping = {
        "automl": "AutoML-GSL",
        "Full": "CEAgent-GSL (Full Method)",
        "baseline": "CEAgent-GSL-Zero",
    }
    return mapping.get(method, method)


def generate_result_tables(input_csv: str, output_dir: str = None):
    """
    生成结果表格
    
    Args:
        input_csv: combined_all_results.csv 路径
        output_dir: 输出的目录路径
    """
    # 读取数据
    data_by_alpha = {}  # {alpha: {folder_name: {method: score}}}
    
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            alpha = row['alpha']
            folder_name = row['folder_name']
            method = row['method']
            score = float(row['test_combined_score'])
            
            if alpha not in data_by_alpha:
                data_by_alpha[alpha] = {}
            
            if folder_name not in data_by_alpha[alpha]:
                data_by_alpha[alpha][folder_name] = {}
            
            data_by_alpha[alpha][folder_name][method] = score
    
    # Method 排列顺序（按用户指定）
    method_order = METHOD_ORDER
    
    # 获取所有 alpha 值并排序（降序），排除 alpha=0.3
    sorted_alphas = sorted(
        [a for a in data_by_alpha.keys() if a != "0.3"],
        key=lambda x: float(x),
        reverse=True
    )
    
    # 获取输出目录
    if output_dir is None:
        output_dir = Path(input_csv).parent.resolve()
    else:
        output_dir = Path(output_dir).resolve()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成合并的大表格
    all_rows = []
    header = ["Method"] + [f"α={alpha}" for alpha in sorted_alphas]
    
    for folder_name in method_order:
        # 检查是否在数据中存在
        if folder_name not in data_by_alpha.get(sorted_alphas[0], {}):
            continue
        
        row = {
            "Method": map_folder_name(folder_name)
        }
        
        for alpha in sorted_alphas:
            # 获取该 alpha 下该方法的所有 method 分数
            folder_data = data_by_alpha.get(alpha, {}).get(folder_name, {})
            
            automl_score = folder_data.get("automl", None)
            full_score = folder_data.get("Full", None)
            baseline_score = folder_data.get("baseline", None)
            
            # 选择最佳分数（优先 automl，然后是 Full，最后是 baseline）
            best_score = None
            if automl_score is not None:
                best_score = automl_score
            elif full_score is not None:
                best_score = full_score
            elif baseline_score is not None:
                best_score = baseline_score
            
            row[f"α={alpha}"] = format_score(best_score)
        
        all_rows.append(row)
    
    # 保存合并表格
    pivot_output = output_dir / "combined_all_results_pivot.csv"
    
    with open(pivot_output, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(all_rows)
    
    print(f"合并表格已保存到: {pivot_output}")
    print(f"\n{'='*60}")
    print("合并表格预览:")
    print(f"{'Method':<18}", end="")
    for alpha in sorted_alphas:
        print(f"{'α='+alpha:>12}", end="")
    print()
    print("-" * (18 + 12 * len(sorted_alphas)))
    for row in all_rows:
        print(f"{row['Method']:<18}", end="")
        for alpha in sorted_alphas:
            print(f"{row[f'α={alpha}']:>12}", end="")
        print()
    
    # 生成详细表格（包含所有方法类型的分数）
    detail_output = output_dir / "combined_all_results_detail.csv"
    
    detail_rows = []
    
    for folder_name in method_order:
        # 检查是否在数据中存在
        if folder_name not in data_by_alpha.get(sorted_alphas[0], {}):
            continue
        
        folder_display = map_folder_name(folder_name)
        
        for alpha in sorted_alphas:
            folder_data = data_by_alpha.get(alpha, {}).get(folder_name, {})
            
            automl_score = folder_data.get("automl", None)
            full_score = folder_data.get("Full", None)
            baseline_score = folder_data.get("baseline", None)
            
            detail_rows.append({
                "alpha": alpha,
                "Method": folder_display,
                "AutoML-GSL": format_score(automl_score),
                "CEAgent-GSL (Full Method)": format_score(full_score),
                "CEAgent-GSL-Zero": format_score(baseline_score)
            })
    
    detail_header = ["alpha", "Method", "CEAgent-GSL (Full Method)", "CEAgent-GSL-Zero", "AutoML-GSL"]
    
    with open(detail_output, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=detail_header)
        writer.writeheader()
        writer.writerows(detail_rows)
    
    print(f"\n详细表格已保存到: {detail_output}")
    print(f"\n{'='*60}")
    print("详细表格预览（前 15 行）:")
    print(f"{'alpha':<8}{'Method':<18}{'AutoML-GSL':>14}{'CEAgent-GSL (Full Method)':>28}{'CEAgent-GSL-Zero':>20}")
    print("-" * 90)
    for row in detail_rows[:15]:
        print(f"{row['alpha']:<8}{row['Method']:<18}{row['AutoML-GSL']:>14}{row['CEAgent-GSL (Full Method)']:>28}{row['CEAgent-GSL-Zero']:>20}")
    
    # 按 alpha 拆分详细表格（排除 alpha=0.3）
    print(f"\n{'='*60}")
    print("按 alpha 拆分详细表格...")
    
    for alpha in sorted_alphas:
        alpha_rows = [r for r in detail_rows if r['alpha'] == alpha]
        alpha_output = output_dir / f"results_alpha{alpha}.csv"
        
        # 不保存 alpha 列
        alpha_header = ["Method", "CEAgent-GSL (Full Method)", "CEAgent-GSL-Zero", "AutoML-GSL"]
        alpha_rows_clean = [
            {k: v for k, v in r.items() if k != 'alpha'} 
            for r in alpha_rows
        ]
        
        with open(alpha_output, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=alpha_header)
            writer.writeheader()
            writer.writerows(alpha_rows_clean)
        
        print(f"  ✓ {alpha_output.name} ({len(alpha_rows)} 行)")
    
    print(f"\n所有文件已保存到: {output_dir}")
    
    return all_rows, detail_rows


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="生成实验结果表格")
    parser.add_argument('--input', type=str, 
                       default='combined_all_results.csv',
                       help='输入的 combined_all_results.csv 文件路径')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='输出的目录路径（可选）')
    
    args = parser.parse_args()
    
    # 使用脚本所在目录
    script_dir = Path(__file__).parent.resolve()
    input_csv = script_dir / args.input
    
    if not input_csv.exists():
        print(f"错误: 文件 {input_csv} 不存在")
        exit(1)
    
    generate_result_tables(str(input_csv), args.output_dir)
