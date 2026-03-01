import os
import json
from openai import OpenAI
import numpy as np
# 初始化客户端（建议放在全局，避免每次调用函数都重新初始化）
api_key="sk-92265d8b2c044989b10ad59e3a27b56f"
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
client = OpenAI(
    api_key=api_key,
    base_url=base_url
)

import numpy as np

def get_embeddings_safe(client, text, model="text-embedding-v4", dimensions=1024):
    if not text:
        return [0.0] * dimensions

    # 【优化1】调整切片大小
    # text-embedding-v4 支持约 8192 tokens。
    # 6000 字符通常是安全的，但为了更严谨，可以保留此设置或略微增加。
    chunk_size = 6000 
    
    # 生成切片
    raw_chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    # 过滤空切片，并记录每个切片的“权重”（这里使用字符长度作为权重）
    valid_chunks = []
    weights = []
    for chunk in raw_chunks:
        clean_chunk = chunk.strip()
        if clean_chunk:
            valid_chunks.append(clean_chunk)
            # 【核心改动A】记录权重：使用文本长度（字符数）
            # 也可以考虑使用 len(chunk.encode('utf-8')) 或估算 token 数，但字符长度通常足够有效
            weights.append(len(clean_chunk))

    if not valid_chunks:
        return [0.0] * dimensions

    all_embeddings = []
    
    # 【优化2】使用 Batch API 调用，而不是单条循环
    # 阿里云百炼 embedding API 通常限制单次请求最多包含 10-25 条输入（v4通常为10）
    batch_limit = 10
    
    for i in range(0, len(valid_chunks), batch_limit):
        batch_input = valid_chunks[i : i + batch_limit]
        
        try:
            # 一次发送一批文本
            resp = client.embeddings.create(
                model=model,
                input=batch_input,
                dimensions=dimensions
            )
            
            # API 返回的 embeddings 顺序与 input 一致
            # 提取 data 中的 embedding 向量
            # 注意：需确保 resp.data 按照 index 排序（通常 API 会保证，但严谨起见可按 index 排序）
            batch_data = sorted(resp.data, key=lambda x: x.index)
            for item in batch_data:
                all_embeddings.append(item.embedding)
                
        except Exception as e:
            print(f"处理 Batch {i//batch_limit + 1} 时出错: {e}")
            # 如果出错，为了保持对齐，需要填充 0 向量或者移除对应的权重？
            # 简单起见，这里跳过，但需同时移除对应的权重以防计算错误
            # (复杂的生产环境代码需要更细致的错误恢复，这里做截断处理)
            current_batch_len = len(batch_input)
            # 移除这部分的权重，因为没有对应的向量
            del weights[len(all_embeddings) : len(all_embeddings) + current_batch_len]

    if not all_embeddings:
        return [0.0] * dimensions

    # 转换为 numpy 数组以便计算
    embeddings_matrix = np.array(all_embeddings)
    
    # 确保权重数量和向量数量一致（以防 API 失败导致的不一致）
    current_weights = weights[:len(embeddings_matrix)]
    
    # 【核心改动B】使用“加权平均”代替“简单平均”
    # weights=current_weights 让长切片对最终向量贡献更大
    final_embedding = np.average(embeddings_matrix, axis=0, weights=current_weights)
    
    # 归一化（可选，但在 Embedding 任务中通常推荐，尤其是加权平均后）
    # 许多向量数据库使用余弦相似度，归一化可以保证点积等价于余弦相似度
    norm = np.linalg.norm(final_embedding)
    if norm > 0:
        final_embedding = final_embedding / norm

    return final_embedding.tolist()

# --- 使用示例 ---
ALL_METHODS = [
        "cta_scdeepsort", "cta_scheteronet", "domain_efnst", "domain_louvain",
        "domain_spagcn", "domain_stagate", "cta_scgat", "cta_scrgcl",
        "cta_graphcs", "domain_stlearn"
    ]
for method_name in ALL_METHODS:
    base_path="/mnt/nfs/zyxing/msu/dance_temp/dance/examples/search/generate_pseudocode/"
    file_path=os.path.join(base_path,f"{method_name}_openevolve_output/best/best_program.txt")

    with open(file_path,"r") as f:
        content=f.read()
    content=content.split('#Pseudocode-START')[1].split('#Pseudocode-END')[0]
    # 2. 调用函数并将结果“存”在变量中
    vector_result = get_embeddings_safe(client,content)

    if vector_result:
        print(f"获取成功，向量维度: {len(vector_result)}")
        print(f"前5位数值: {vector_result[:5]}")
    np.save(f"data/{method_name}_embedding.npy",np.array(vector_result))