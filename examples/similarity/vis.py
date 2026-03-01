import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

path = "similarity_matrix.csv"

# 1. 读取数据
df = pd.read_csv(path, index_col=0)

# ================= 修改开始 =================

# 2. 定义映射字典
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

# 3. 应用重命名
# 使用 rename 方法同时修改索引(index)和列名(columns)
df = df.rename(index=label_mapping, columns=label_mapping)

# (可选) 4. 重新排序
# 如果你希望热图的顺序严格对应你提供的 LaTeX 表格顺序，请取消下面两行的注释：
# desired_order = ['GraphCS', 'ScDeepSort', 'scGAT', 'scHeteroNet', 'scRGCL', 'EfNST', 'Louvain', 'SpaGCN', 'spaGRA', 'STAGATE', 'stLearn']
# df = df.loc[desired_order, desired_order]

# ================= 修改结束 =================

# 5. 处理对角线
np.fill_diagonal(df.values, 0)

# 6. 设置绘图风格
sns.set(font_scale=1.0) # 建议稍微调大一点，比如 1.1 或 1.2，适应论文阅读
plt.figure(figsize=(12, 10))

# 7. 绘制热度图
ax = sns.heatmap(df, cmap="Blues", annot=True, fmt=".2f", 
                 cbar_kws={'label': 'Semantic Similarity Score'},
                 linewidths=1, linecolor='white')

# 8. 在每行的最大值处画红框
for i, row_name in enumerate(df.index):
    max_val = df.loc[row_name].max()
    # 防止全0行报错（虽然不太可能）
    if max_val > 0: 
        max_cols = df.columns[df.loc[row_name] == max_val].tolist()
        
        for col_name in max_cols:
            j = df.columns.get_loc(col_name)
            # 添加红框
            ax.add_patch(patches.Rectangle((j, i), 1, 1, fill=False, edgecolor='red', lw=3, clip_on=False))

# 9. 调整标签
plt.title("Model-to-Model Similarity Landscape & Retrieval Trajectory", fontsize=14, pad=20)
plt.xlabel("Reference Model (Historical Knowledge)", fontsize=12)
plt.ylabel("Target Model (Current Task)", fontsize=12)

# 旋转 X 轴标签以防重叠，且对齐
plt.xticks(rotation=45, ha='right') 
plt.yticks(rotation=0)

plt.tight_layout()

# 保存
plt.savefig("model_similarity_heatmap.pdf", dpi=300, bbox_inches='tight')
plt.show()