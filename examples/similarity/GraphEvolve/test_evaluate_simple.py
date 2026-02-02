
#!/usr/bin/env python3
"""
测试 LamarckianKnowledgeBase 的完整工作流程

基于 eigenvectors_complex 例子测试：
1. 初始化知识库
2. 检索知识（第一次应该为空）
3. 学习轨迹（learn_from_trajectory）
4. 再次检索知识（应该能检索到刚学习的知识）
"""

import sys
import os
import shutil
import asyncio

# 添加项目路径到 sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from lamarckian_knowledge_base import LamarckianKnowledgeBase
task_query = "cta_scdeepsort"
# 设置路径
example_dir = "/mnt/nfs/zyxing/msu/dance_temp/dance/examples/evolo"
evaluator_file = os.path.join(example_dir, "evaluator.py")
initial_program_file = os.path.join(example_dir,task_query,"initial_program.py")
best_program_file = os.path.join(example_dir,task_query, "openevolve_output", "best", "best_program.py")
config_yaml_path=os.path.join(example_dir,task_query,'config.yaml')

# # 设置路径
# example_dir = "/mnt/nfs/zyxing/msu/dance_temp/dance/examples/similarity/GraphEvolve/openevolve/examples/algotune/eigenvectors_complex"
# evaluator_file = os.path.join(example_dir, "evaluator.py")
# initial_program_file = os.path.join(example_dir,"initial_program.py")
# best_program_file = os.path.join(example_dir, "best_program.py")
# config_yaml_path=os.path.join(example_dir,'config.yaml')

    # 初始化知识库
test_db_path = "./test_db4"
print(f"\n{'='*80}")
print("初始化知识库")
print(f"{'='*80}")
print(f"向量数据库路径: {test_db_path}")
try:
    kb = LamarckianKnowledgeBase(
        vector_store_path=test_db_path
    )
    print("✅ 知识库初始化成功")
except Exception as e:
    print(f"❌ 知识库初始化失败: {e}")
    sys.exit(1)
def test_full_workflow():
    """测试完整的工作流程"""
    
    print("=" * 80)
    print("测试 LamarckianKnowledgeBase 完整工作流程")
    print("=" * 80)
    
    # 检查 API key
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key or api_key == "YOUR_DASHSCOPE_API_KEY":
        print("\n⚠️  警告: 未设置 DASHSCOPE_API_KEY 环境变量")
        print("   请设置环境变量: export DASHSCOPE_API_KEY='your-api-key'")
        print("   或者测试将在 LLM 调用时失败")
        print()
        
  
    # ============================================
    print(f"\n{'='*80}")
    print("步骤 0: 根据 task_query 删除检索到的规则")
    print(f"{'='*80}")

    try:
        # 使用 task_query 检索知识
        retrieved = kb.retrieve_knowledge(task_query, k=10)
        print(f"\n查询: {task_query}")
        print(f"检索到 {len(retrieved['principles'])} 个原则, {len(retrieved['trajectories'])} 个轨迹")

        # 获取所有检索到的文档的 ID
        all_memories = kb.list_all_memories()
        principles = all_memories.get("principles", [])
        trajectories = all_memories.get("trajectories", [])

        # 筛选出与 task_query 相关的规则（通过比较内容是否在检索结果中）
        ids_to_delete = []

        for p in principles:
            if p.get("content") in retrieved['principles']:
                print(p.get("content"))
                if p.get("id"):
                    ids_to_delete.append(p["id"])

        for t in trajectories:
            if t.get("content") in retrieved['trajectories']:
                print(t.get("content"))
                if t.get("id"):
                    ids_to_delete.append(t["id"])

        # 删除检索到的规则
        if ids_to_delete:
            print(f"\n删除 {len(ids_to_delete)} 条检索到的规则...")
            # 调用底层 collection 的 delete 方法
            collection = getattr(kb.vector_store, "_collection", None)
            if collection and hasattr(collection, "delete"):
                collection.delete(ids=ids_to_delete)
                print(f"✅ 已删除 {len(ids_to_delete)} 条规则")
            else:
                print("⚠️  无法访问底层 collection，无法删除规则")
        else:
            print("没有需要删除的规则")

    except Exception as e:
        print(f"⚠️  删除规则过程出现错误（不影响继续执行）: {e}")
        import traceback
        traceback.print_exc()
    
    # ============================================
    # 步骤 2: 检索知识（第一次，应该为空）
    # ============================================
    print(f"\n{'='*80}")
    print("步骤 2: 检索知识（第一次）")
    print(f"{'='*80}")
    
    
    try:
        retrieved = kb.retrieve_knowledge(task_query, k=3)
        print(f"\n查询: {task_query}")
        print(f"\n检索结果:")
        print(f"  - 原则数量: {len(retrieved['principles'])}")
        print(f"  - 轨迹数量: {len(retrieved['trajectories'])}")
        
        if retrieved['principles']:
            print(f"\n找到的原则:")
            for i, principle in enumerate(retrieved['principles'], 1):
                print(f"  {i}. {principle[:100]}...")
        else:
            print("  (没有找到相关原则，这是正常的，因为知识库是空的)")
        
        if retrieved['trajectories']:
            print(f"\n找到的轨迹:")
            for i, trajectory in enumerate(retrieved['trajectories'], 1):
                print(f"  {i}. {trajectory[:100]}...")
        else:
            print("  (没有找到相关轨迹，这是正常的，因为知识库是空的)")
        
        print("✅ 检索完成")
    except Exception as e:
        print(f"❌ 检索失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ============================================
    # 步骤 3: 学习轨迹（learn_from_trajectory）
    # ============================================
    print(f"\n{'='*80}")
    print("步骤 3: 学习轨迹（learn_from_trajectory）")
    print(f"{'='*80}")
    print(f"初始程序: {initial_program_file}")
    print(f"最优程序: {best_program_file}")
    
    # 检查文件是否存在
    if not os.path.exists(initial_program_file):
        print(f"❌ 初始程序文件不存在: {initial_program_file}")
        return False
    
    if not os.path.exists(best_program_file):
        print(f"❌ 最优程序文件不存在: {best_program_file}")
        return False
    
    try:
        result = asyncio.run(
            kb.learn_from_trajectory(
                initial_program_path=initial_program_file,
                best_program_path=best_program_file,
                original_task=task_query,
                evaluator_file=evaluator_file,  # 传入 evaluator 文件路径
                metrics=None,  # 让系统自动评估 initial_program
                config=config_yaml_path,  # 传入 config.yaml，供 OpenEvolve 使用
            )
        )
        
        print(f"\n学习结果:")
        print(f"  {'-'*70}")
        print(f"  提取的所有原则 ({len(result['all_principles'])} 条):")
        for i, p in enumerate(result['all_principles'], 1):
            print(f"    {i}. {p}")
        print(f"\n  已验证并存储的原则 ({result['saved_count']} 条):")
        if result['extracted_principles']:
            for i, p in enumerate(result['extracted_principles'], 1):
                print(f"    ✅ {i}. {p}")
        else:
            print(f"    (无)")
        print(f"\n  各原则验证详情:")
        for i, r in enumerate(result['results'], 1):
            status = "✅ VERIFIED" if r['saved'] else "❌ REJECTED"
            print(f"    {i}. {status}: {r['principle'][:60]}...")
        print(f"  {'-'*70}")

        if result['saved_count'] > 0:
            print(f"✅ {result['saved_count']} 条原则已验证并保存到知识库")
        else:
            print("⚠️  没有原则通过验证，未保存到知识库")
        
        print("✅ 学习流程完成")
        
        # 列出并打印当前知识库中保存的所有 principle / trajectory（便于快速检查）
        try:
            all_memories = kb.list_all_memories()
            principles = all_memories.get("principles", [])
            trajectories = all_memories.get("trajectories", [])

            print(f"\n=== 当前知识库概览 ===")
            print(f"  - 原则总数: {len(principles)}")
            print(f"  - 轨迹总数: {len(trajectories)}")

            if principles:
                print("\n已保存的原则（前 5 条）：")
                for i, p in enumerate(principles[:5], 1):
                    meta = p.get("metadata", {})
                    content_preview = p.get("content", "")[:300].replace("\n", " ")
                    print(f"  {i}. {content_preview}")
                    print(f"     source_task: {meta.get('source_task')}, id: {p.get('id')}")

            if trajectories:
                print("\n已保存的轨迹（前 3 条）：")
                for i, t in enumerate(trajectories[:3], 1):
                    meta = t.get("metadata", {})
                    traj_preview = t.get("content", "")[:300].replace("\n", " ")
                    print(f"  {i}. {traj_preview}")
                    print(f"     source_task: {meta.get('source_task')}, id: {t.get('id')}")
        except Exception as e:
            print(f"无法列出知识库内容: {e}")
        
    except Exception as e:
        print(f"❌ 学习流程失败: {e}")
        import traceback
        print(f"\n详细错误信息:")
        traceback.print_exc()
        return False
    
    # ============================================
    # 步骤 4: 测试不同的查询（语义相似）
    # ============================================
    print(f"\n{'='*80}")
    print("步骤 4: 测试不同的查询（语义相似）")
    print(f"{'='*80}")
    
    similar_queries = [
        task_query
    ]
    
    for query in similar_queries:
        try:
            retrieved = kb.retrieve_knowledge(query, k=2)
            print(f"\n查询: {query}")
            print(f"  找到 {len(retrieved['principles'])} 个原则, {len(retrieved['trajectories'])} 个轨迹")
            
            if retrieved['principles']:
                print(f"  原则预览: {retrieved['principles'][0][:80]}...")
            if retrieved['trajectories']:
                print(f"  轨迹预览: {retrieved['trajectories'][0][:80]}...")
        except Exception as e:
            print(f"❌ 查询 '{query}' 失败: {e}")
    
    print("\n✅ 所有测试完成！")
    return True


def test_retrieve_different_queries():
    """
    测试不同的查询是否能检索到之前学习的知识
    
    Args:
        kb: 已初始化的知识库实例（应该已经存储了一些知识）
    """
    print(f"\n{'='*80}")
    print("测试: 不同的语义查询")
    print(f"{'='*80}")
    
    # 不同的查询方式，验证向量搜索的语义匹配能力
    test_queries = [
        # 中文同义表达
        ("中文查询1", task_query),
        ("中文查询2", "如何计算矩阵的特征向量"),
        ("中文查询3", "矩阵特征值分解"),
        
        # 英文表达
        ("英文查询1", "eigenvalue eigenvector computation"),
        ("英文查询2", "matrix eigenvalue decomposition"),
        ("英文查询3", "compute eigenvectors of a matrix"),
    ]
    
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
                print(f"  原则: {retrieved['principles'][0][:60]}...")
            if retrieved['trajectories']:
                print(f"  轨迹: {retrieved['trajectories'][0][:60]}...")
            
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
    success = test_full_workflow()
    sys.exit(0 if success else 1)

