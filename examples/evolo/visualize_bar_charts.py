import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置matplotlib参数
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150

# 读取CSV文件
csv_path = "/mnt/nfs/zyxing/msu/dance_temp/dance/examples/evolo/all_combined_scores.csv"
df = pd.read_csv(csv_path)

# 将is_evoloved列转换为布尔类型
df['is_evoloved'] = df['is_evoloved'].astype(str).str.lower() == 'true'

# 过滤掉alpha=0.3的数据
df = df[df['alpha'] != 0.3]
print("已过滤掉alpha=0.3的数据")

# 方法名称映射（原始名称 -> 正式名称）
label_mapping = {
    'cta_graphcs': 'GraphCS',
    'cta_scdeepsort': 'ScDeepSort',
    'cta_scgat': 'scGAT',
    'cta_scheteronet': 'scHeteroNet',
    'cta_scrgcl': 'scRGCL',
    'domain_efnst': 'EfNST',
    'domain_louvain': 'Louvain',
    'domain_spagcn': 'SpaGCN',
    'domain_spagra': 'spaGRA',
    'domain_stagate': 'STAGATE',
    'domain_stlearn': 'stLearn'
}

# 获取所有方法名称
methods = sorted(df['folder_name'].unique())

# 获取唯一的alpha值（排序）
alphas = sorted(df['alpha'].unique())

# 定义颜色
colors = {False: '#2E86AB', True: '#E94F37'}

# 创建输出目录
output_dir = "/mnt/nfs/zyxing/msu/dance_temp/dance/examples/evolo/bar_charts"
os.makedirs(output_dir, exist_ok=True)

# ==========================================
# 1. 为每个方法生成单独的小图
# ==========================================
print("生成单独的方法对比图...")

for method in methods:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 筛选该方法的数据
    method_data = df[df['folder_name'] == method]
    
    # 获取该方法在所有alpha下的数据
    x_positions = np.arange(len(alphas))
    width = 0.35
    
    # 获取正式名称
    display_name = label_mapping.get(method, method)
    
    # 分别存储两种状态的分数（使用test_combined_score，这是真正随alpha变化的组合得分）
    # CEAgent-GSL-Zero: Not Evolved (消融实验)
    # CEAgent-GSL: Evolved (我们的方法)
    
    scores_zero = []
    scores_gs = []
    
    for alpha in alphas:
        # CEAgent-GSL-Zero 数据
        zero_data = method_data[(method_data['alpha'] == alpha) & (method_data['is_evoloved'] == False)]
        if not zero_data.empty:
            scores_zero.append(zero_data['test_combined_score'].values[0])
        else:
            scores_zero.append(0)
        
        # CEAgent-GSL 数据
        gs_data = method_data[(method_data['alpha'] == alpha) & (method_data['is_evoloved'] == True)]
        if not gs_data.empty:
            scores_gs.append(gs_data['test_combined_score'].values[0])
        else:
            scores_gs.append(0)
    
    # 绘制 CEAgent-GSL-Zero 柱子
    bars1 = ax.bar(x_positions - width/2, scores_zero, width, 
                   color=colors[False],
                   label='CEAgent-GSL-Zero',
                   alpha=0.85,
                   edgecolor='white',
                   linewidth=1)
    
    # 在 CEAgent-GSL-Zero 柱子上添加数值标签
    for bar, score in zip(bars1, scores_zero):
        if score > 0:
            ax.annotate(f'{score:.3f}',
                       (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                       textcoords="offset points",
                       xytext=(0, 5),
                       ha='center',
                       va='bottom',
                       fontsize=9,
                       fontweight='bold',
                       color=colors[False])
    
    # 绘制 CEAgent-GSL 柱子
    bars2 = ax.bar(x_positions + width/2, scores_gs, width, 
                   color=colors[True],
                   label='CEAgent-GSL',
                   alpha=0.85,
                   edgecolor='white',
                   linewidth=1)
    
    # 在 CEAgent-GSL 柱子上添加数值标签
    for bar, score in zip(bars2, scores_gs):
        if score > 0:
            ax.annotate(f'{score:.3f}',
                       (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                       textcoords="offset points",
                       xytext=(0, 5),
                       ha='center',
                       va='bottom',
                       fontsize=9,
                       fontweight='bold',
                       color=colors[True])
    
    # 设置图表属性
    ax.set_xlabel('Alpha', fontsize=12, fontweight='bold')
    ax.set_ylabel('Combined Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Combined Score Comparison for {display_name}\n(CEAgent-GSL-Zero vs CEAgent-GSL across Alpha values)', 
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(a) for a in alphas], fontsize=11)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    
    # 计算y轴范围（从数据的最小值开始，留出边距）
    all_scores = scores_zero + scores_gs
    all_scores = [s for s in all_scores if s > 0]
    if all_scores:
        y_min = min(all_scores) * 0.9
        y_max = max(all_scores) * 1.15
        ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    
    # 保存单独的图片
    safe_method_name = method.replace('/', '_')
    output_file = os.path.join(output_dir, f'{safe_method_name}_comparison.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  已保存: {output_file}")

print(f"\n完成！共生成 {len(methods)} 个单独的方法对比图")

# ==========================================
# 2. 生成大汇总图（所有方法在一个图中）
# ==========================================
print("\n生成大汇总图...")

# 计算需要的行列数
n_methods = len(methods)
n_cols = 3
n_rows = int(np.ceil(n_methods / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
fig.suptitle('Combined Score Comparison Across All Methods\n(CEAgent-GSL-Zero vs CEAgent-GSL across Alpha values)', 
             fontsize=16, fontweight='bold', y=1.02)

# 展平axes数组以便迭代
axes_flat = axes.flatten()

for idx, method in enumerate(methods):
    ax = axes_flat[idx]
    
    # 筛选该方法的数据
    method_data = df[df['folder_name'] == method]
    
    x_positions = np.arange(len(alphas))
    width = 0.35
    
    # 分别存储两种状态的分数（使用test_combined_score）
    # CEAgent-GSL-Zero: 消融实验
    # CEAgent-GSL: 我们的方法
    scores_zero = []
    scores_gs = []
    
    for alpha in alphas:
        zero_data = method_data[(method_data['alpha'] == alpha) & (method_data['is_evoloved'] == False)]
        if not zero_data.empty:
            scores_zero.append(zero_data['test_combined_score'].values[0])
        else:
            scores_zero.append(0)
        
        gs_data = method_data[(method_data['alpha'] == alpha) & (method_data['is_evoloved'] == True)]
        if not gs_data.empty:
            scores_gs.append(gs_data['test_combined_score'].values[0])
        else:
            scores_gs.append(0)
    
    # 绘制两组柱子
    bars1 = ax.bar(x_positions - width/2, scores_zero, width, 
                   color=colors[False],
                   label='CEAgent-GSL-Zero',
                   alpha=0.85)
    
    bars2 = ax.bar(x_positions + width/2, scores_gs, width, 
                   color=colors[True],
                   label='CEAgent-GSL',
                   alpha=0.85)
    
    # 简化标签
    for bar, score in zip(bars1, scores_zero):
        if score > 0:
            ax.annotate(f'{score:.2f}',
                       (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                       textcoords="offset points",
                       xytext=(0, 3),
                       ha='center',
                       va='bottom',
                       fontsize=7,
                       color=colors[False])
    
    for bar, score in zip(bars2, scores_gs):
        if score > 0:
            ax.annotate(f'{score:.2f}',
                       (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                       textcoords="offset points",
                       xytext=(0, 3),
                       ha='center',
                       va='bottom',
                       fontsize=7,
                       color=colors[True])
    
    ax.set_title(label_mapping.get(method, method), fontsize=11, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(a) for a in alphas], fontsize=9)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    
    # 每个子图有独立的y轴范围
    all_scores = scores_zero + scores_gs
    all_scores = [s for s in all_scores if s > 0]
    if all_scores:
        y_min = min(all_scores) * 0.9
        y_max = max(all_scores) * 1.2
        ax.set_ylim(y_min, y_max)
    
    # 只在左侧显示y轴标签
    if idx % n_cols == 0:
        ax.set_ylabel('Score', fontsize=10)
    
    # 只在最后一行显示x轴标签
    if idx >= (n_rows - 1) * n_cols:
        ax.set_xlabel('Alpha', fontsize=10)
    
    # 只在第一个子图显示图例
    if idx == 0:
        ax.legend(loc='upper left', fontsize=8)

# 隐藏多余的子图
for idx in range(n_methods, len(axes_flat)):
    axes_flat[idx].set_visible(False)

plt.tight_layout()

# 保存大汇总图
summary_output = os.path.join(output_dir, 'all_methods_summary.png')
plt.savefig(summary_output, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print(f"已保存大汇总图: {summary_output}")

# ==========================================
# 3. 生成按方法类型分组的大图（CTA vs Domain）
# ==========================================
print("\n生成按类型分组的大图...")

# 分离CTA和Domain方法
cta_methods = [m for m in methods if m.startswith('cta_')]
domain_methods = [m for m in methods if m.startswith('domain_')]

for method_type, type_methods in [('CTA', cta_methods), ('Domain', domain_methods)]:
    n_type = len(type_methods)
    n_cols = 2
    n_rows = int(np.ceil(n_type / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    fig.suptitle(f'{method_type} Methods Combined Score Comparison\n(CEAgent-GSL-Zero vs CEAgent-GSL across Alpha values)', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    axes_flat = axes.flatten()
    
    for idx, method in enumerate(type_methods):
        ax = axes_flat[idx]
        
        method_data = df[df['folder_name'] == method]
        
        x_positions = np.arange(len(alphas))
        width = 0.35
        
        # 分别存储两种状态的分数（使用test_combined_score）
        # CEAgent-GSL-Zero: 消融实验
        # CEAgent-GSL: 我们的方法
        scores_zero = []
        scores_gs = []
        
        for alpha in alphas:
            zero_data = method_data[(method_data['alpha'] == alpha) & (method_data['is_evoloved'] == False)]
            if not zero_data.empty:
                scores_zero.append(zero_data['test_combined_score'].values[0])
            else:
                scores_zero.append(0)
            
            gs_data = method_data[(method_data['alpha'] == alpha) & (method_data['is_evoloved'] == True)]
            if not gs_data.empty:
                scores_gs.append(gs_data['test_combined_score'].values[0])
            else:
                scores_gs.append(0)
        
        bars1 = ax.bar(x_positions - width/2, scores_zero, width, 
                       color=colors[False],
                       label='CEAgent-GSL-Zero',
                       alpha=0.85)
        
        bars2 = ax.bar(x_positions + width/2, scores_gs, width, 
                       color=colors[True],
                       label='CEAgent-GSL',
                       alpha=0.85)
        
        for bar, score in zip(bars1, scores_zero):
            if score > 0:
                ax.annotate(f'{score:.2f}',
                           (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                           textcoords="offset points",
                           xytext=(0, 3),
                           ha='center',
                           va='bottom',
                           fontsize=8,
                           color=colors[False])
        
        for bar, score in zip(bars2, scores_gs):
            if score > 0:
                ax.annotate(f'{score:.2f}',
                           (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                           textcoords="offset points",
                           xytext=(0, 3),
                           ha='center',
                           va='bottom',
                           fontsize=8,
                           color=colors[True])
        
        ax.set_title(label_mapping.get(method, method), fontsize=11, fontweight='bold')
        ax.set_xticks(x_positions)
        ax.set_xticklabels([str(a) for a in alphas], fontsize=9)
        ax.grid(True, axis='y', linestyle='--', alpha=0.4)
        
        # 每个子图有独立的y轴范围
        all_scores = scores_zero + scores_gs
        all_scores = [s for s in all_scores if s > 0]
        if all_scores:
            y_min = min(all_scores) * 0.9
            y_max = max(all_scores) * 1.2
            ax.set_ylim(y_min, y_max)
        
        if idx % n_cols == 0:
            ax.set_ylabel('Score', fontsize=10)
        
        if idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel('Alpha', fontsize=10)
        
        if idx == 0:
            ax.legend(loc='upper left', fontsize=8)
    
    # 隐藏多余的子图
    for idx in range(n_type, len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    plt.tight_layout()
    
    type_output = os.path.join(output_dir, f'{method_type.lower()}_methods_summary.png')
    plt.savefig(type_output, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  已保存: {type_output}")

print(f"\n{'='*60}")
print(f"所有图表已保存到: {output_dir}")
print(f"{'='*60}")
print(f"生成的文件:")
print(f"  - {len(methods)} 个单独方法对比图")
print(f"  - 1 个全部方法汇总大图 (all_methods_summary.png)")
print(f"  - 2 个按类型分组的大图 (cta_methods_summary.png, domain_methods_summary.png)")
print(f"  共 {len(methods) + 3} 个图表文件")
