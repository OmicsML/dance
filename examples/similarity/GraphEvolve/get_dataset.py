#!/usr/bin/env python3
import sys
from lamarckian_knowledge_base import LamarckianKnowledgeBase

# ================= 配置区域 =================
SERVER_HOST = "211.87.232.112"
SERVER_PORT = 8000

print(f"\n{'='*80}")
print("初始化知识库 (Client-Server 模式)")
print(f"{'='*80}")
print(f"Server: http://{SERVER_HOST}:{SERVER_PORT}")

try:
    # 🌟 核心修改：使用 host 和 port 初始化，不再使用本地路径
    kb = LamarckianKnowledgeBase(
        host=SERVER_HOST,
        port=SERVER_PORT
    )
    print("✅ 知识库连接成功")
except Exception as e:
    print(f"❌ 知识库连接失败: {e}")
    sys.exit(1)


def test_retrieve_different_queries():
    """
    测试不同的查询是否能检索到之前学习的知识
    """
    print(f"\n{'='*80}")
    print("测试: 不同的语义查询")
    print(f"{'='*80}")
    
    # 不同的查询方式，验证向量搜索的语义匹配能力
    # 注意：这里的 query 最好是和之前存入数据相关的关键词
    test_queries = [("英文查询1", "IF performing PCA on high-dimensional single-cell spatial omics features THEN")]
    
    all_passed = True
    results_summary = []
    
    for name, query in test_queries:
        try:
            retrieved = kb.retrieve_knowledge(query, k=2)
            
            has_principles = len(retrieved['principles']) > 0
            has_trajectories = len(retrieved['trajectories']) > 0
            success = has_principles or has_trajectories
            
            status = "✅ 成功" if success else "❌ 失败"
            print(f"\n{name}: {status}")
            print(f"  查询: {query}")
            print(f"  找到 {len(retrieved['principles'])} 个原则, {len(retrieved['trajectories'])} 个轨迹")
            
            if retrieved['principles']:
                print(f"  原则示例: {retrieved['principles'][0][:60]}...")
            if retrieved['trajectories']:
                print(f"  轨迹示例: {retrieved['trajectories'][0][:60]}...")
            
            results_summary.append({
                "name": name,
                "success": success,
                "principles_count": len(retrieved['principles']),
                "trajectories_count": len(retrieved['trajectories'])
            })
            
            if not success:
                all_passed = False
                
        except Exception as e:
            print(f"\n{name}: ❌ 异常")
            print(f"  查询: {query}")
            print(f"  错误: {e}")
            results_summary.append({
                "name": name,
                "success": False,
                "error": str(e)
            })
            all_passed = False
    
    # 汇总结果
    print(f"\n{'='*80}")
    print("测试结果汇总")
    print(f"{'='*80}")
    
    passed = sum(1 for r in results_summary if r.get("success", False))
    total = len(results_summary)
    
    print(f"通过: {passed}/{total}")
    
    for r in results_summary:
        status = "✅" if r.get("success") else "❌"
        print(f"  {status} {r['name']}: {r.get('principles_count', 0)} 原则, {r.get('trajectories_count', 0)} 轨迹")
    
    return all_passed


if __name__ == "__main__":
    # 1. 打印当前知识库的所有内容概览
    print("\n=== 正在获取当前数据库中的所有记忆... ===")
    try:
        memories = kb.list_all_memories()
        
        # 简单统计
        p_list = memories.get('principles', [])
        t_list = memories.get('trajectories', [])
        tasks=[]
        for p in p_list:
            task=p.get("metadata", {}).get("source_task")
            tasks.append(task)
            print(task )
        print(f"📊 统计: 原则 {len(p_list)} 条, 轨迹 {len(t_list)} 条")
        print(f"📊 统计: 任务 {set(tasks)} 个")
        
        # 如果你想看详细内容，可以将下面的注释取消，但内容可能很多
        # import json
        # print(json.dumps(memories, default=str, indent=2, ensure_ascii=False))

    except Exception as e:
        print(f"❌ 获取列表失败: {e}")

    # # 2. 运行检索测试
    # print("\n=== 开始检索测试 ===")
    # test_retrieve_different_queries()