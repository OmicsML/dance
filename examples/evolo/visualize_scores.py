import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 读取CSV文件
csv_path = "/mnt/nfs/zyxing/msu/dance_temp/dance/examples/evolo/all_combined_scores.csv"
df = pd.read_csv(csv_path)

# 将is_evoloved列转换为布尔类型
df['is_evoloved'] = df['is_evoloved'].astype(str).str.lower() == 'true'

# 获取唯一的alpha值
alphas = sorted(df['alpha'].unique())

# 获取所有方法名称（按原始顺序）
methods = df['folder_name'].unique().tolist()

# 定义颜色和标记
colors = {False: '#2E86AB', True: '#E94F37'}
markers = {False: 'o', True: 's'}
linestyles = {False: '-', True: '--'}

# 创建输出目录
output_dir = "/mnt/nfs/zyxing/msu/dance_temp/dance/examples/evolo/line_charts"
os.makedirs(output_dir, exist_ok=True)

# 为每个alpha值创建单独的图
for alpha in alphas:
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # 筛选当前alpha的数据
    alpha_data = df[df['alpha'] == alpha]
    
    # 获取该alpha下的所有方法
    alpha_methods = alpha_data['folder_name'].unique()
    x_positions = np.arange(len(alpha_methods))
    
    # 绘制两条折线
    for evoloved_status in [False, True]:
        # 筛选is_evoloved状态的数据
        subset = alpha_data[alpha_data['is_evoloved'] == evoloved_status]
        
        # 获取按方法顺序排列的数据
        scores = []
        for method in alpha_methods:
            method_data = subset[subset['folder_name'] == method]
            if not method_data.empty:
                scores.append(method_data['combined_score'].values[0])
            else:
                scores.append(np.nan)
        
        # 绘制折线
        label = 'Not Evolved (is_evoloved=False)' if not evoloved_status else 'Evolved (is_evoloved=True)'
        ax.plot(x_positions, scores, 
                color=colors[evoloved_status],
                marker=markers[evoloved_status],
                linestyle=linestyles[evoloved_status],
                linewidth=2.5,
                markersize=10,
                label=label,
                alpha=0.9)
        
        # 在每个点上添加数值标签
        for i, score in enumerate(scores):
            if not np.isnan(score):
                ax.annotate(f'{score:.3f}', 
                           (i, score), 
                           textcoords="offset points",
                           xytext=(0, 10),
                           ha='center',
                           fontsize=8,
                           color=colors[evoloved_status])
    
    # 设置图表属性
    ax.set_xlabel('Method', fontsize=12, fontweight='bold')
    ax.set_ylabel('Combined Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Combined Score Comparison at Alpha = {alpha}\n(Not Evolved vs Evolved)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels(alpha_methods, rotation=45, ha='right', fontsize=10)
    
    # 添加网格
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 添加图例
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    
    # 设置y轴范围，留出一些边距
    y_min = df[df['alpha'] == alpha]['combined_score'].min() - 0.05
    y_max = df[df['alpha'] == alpha]['combined_score'].max() + 0.05
    ax.set_ylim(y_min, y_max)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    output_file = os.path.join(output_dir, f'alpha_{alpha}_combined_score_comparison.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"已保存: {output_file}")

# 创建汇总对比图（所有alpha值在一起）- 2行3列布局
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.flatten()

for idx, alpha in enumerate(alphas):
    ax = axes[idx]
    
    # 筛选当前alpha的数据
    alpha_data = df[df['alpha'] == alpha]
    
    # 获取该alpha下的所有方法
    alpha_methods = alpha_data['folder_name'].unique()
    x_positions = np.arange(len(alpha_methods))
    
    # 绘制两条折线
    for evoloved_status in [False, True]:
        subset = alpha_data[alpha_data['is_evoloved'] == evoloved_status]
        
        scores = []
        for method in alpha_methods:
            method_data = subset[subset['folder_name'] == method]
            if not method_data.empty:
                scores.append(method_data['combined_score'].values[0])
            else:
                scores.append(np.nan)
        
        label = 'Not Evolved' if not evoloved_status else 'Evolved'
        ax.plot(x_positions, scores, 
                color=colors[evoloved_status],
                marker=markers[evoloved_status],
                linestyle=linestyles[evoloved_status],
                linewidth=2,
                markersize=8,
                label=label,
                alpha=0.9)
    
    ax.set_xlabel('Method', fontsize=10)
    ax.set_ylabel('Combined Score', fontsize=10)
    ax.set_title(f'Alpha = {alpha}', fontsize=12, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(alpha_methods, rotation=45, ha='right', fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='best', fontsize=8)

# 隐藏多余的子图（如果有）
for idx in range(len(alphas), len(axes)):
    axes[idx].set_visible(False)

plt.suptitle('Combined Score Comparison Across Different Alpha Values\n(Not Evolved vs Evolved)', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()

# 保存汇总图
summary_output = os.path.join(output_dir, 'all_alphas_comparison.png')
plt.savefig(summary_output, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print(f"\n已保存汇总图: {summary_output}")
print(f"\n所有图表已保存到: {output_dir}")
print(f"共生成 {len(alphas) + 1} 个图表文件")
