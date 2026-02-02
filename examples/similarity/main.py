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

def get_embeddings_safe(client, text, model="text-embedding-v4", dimensions=1024):
    # 简单的按字符分块逻辑
    chunk_size = 6000  
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    # 【改动1】创建一个列表来存放每个块的独立向量，而不是把它们压扁
    chunk_embeddings = []
    
    for chunk in chunks:
        if not chunk.strip(): continue 
        
        try:
            resp = client.embeddings.create(
                model=model,
                input=[chunk],
                dimensions=dimensions
            )
            # 【改动2】append 整个向量对象，而不是 extend 元素
            # 此时 chunk_embeddings 是一个二维列表 [[...1024...], [...1024...]]
            chunk_embeddings.append(resp.data[0].embedding)
        except Exception as e:
            print(f"处理分块时出错: {e}")
            
    # 【改动3】如果没有向量（比如文本为空），返回全0向量
    if not chunk_embeddings:
        return [0.0] * dimensions

    # 【核心修正】计算平均值
    # axis=0 表示沿着“行”的方向压缩，把多个向量平均成一个
    # 结果是一个 (1024,) 的向量
    final_embedding = np.mean(chunk_embeddings, axis=0)
    
    # 转换回列表格式 (如果你后续代码需要 list 而不是 numpy array)
    return final_embedding.tolist()

# --- 使用示例 ---

method_name="cta_scheteronet"

base_path="/mnt/nfs/zyxing/msu/dance_temp/dance/examples/search/generate_pseudocode/"
file_path=os.path.join(base_path,f"{method_name}_openevolve_output/best/best_program.txt")

with open(file_path,"r") as f:
    content=f.read()
# 2. 调用函数并将结果“存”在变量中
vector_result = get_embeddings_safe(client,content)

if vector_result:
    print(f"获取成功，向量维度: {len(vector_result)}")
    print(f"前5位数值: {vector_result[:5]}")
np.save(f"data/{method_name}_embedding.npy",np.array(vector_result))