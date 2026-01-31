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

# 添加项目路径到 sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from lamarckian_knowledge_base import LamarckianKnowledgeBase

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
    
    # 设置路径
    example_dir = "/mnt/nfs/zyxing/msu/dance_temp/dance/examples/similarity/GraphEvolve/openevolve/examples/algotune/eigenvectors_complex"
    evaluator_file = os.path.join(example_dir, "evaluator.py")
    initial_program_file = os.path.join(example_dir, "initial_program.py")
    best_program_file = os.path.join(example_dir, "best_program.py")
    
    # 清理旧的测试数据库
    test_db_path = "./test_workflow_db"
    if os.path.exists(test_db_path):
        print(f"\n清理旧的测试数据库: {test_db_path}")
        shutil.rmtree(test_db_path)
    
    # ============================================
    # 步骤 1: 初始化知识库
    # ============================================
    print(f"\n{'='*80}")
    print("步骤 1: 初始化知识库")
    print(f"{'='*80}")
    print(f"Evaluator 文件: {evaluator_file}")
    print(f"向量数据库路径: {test_db_path}")
    
    try:
        kb = LamarckianKnowledgeBase(
            vector_store_path=test_db_path
        )
        print("✅ 知识库初始化成功")
    except Exception as e:
        print(f"❌ 知识库初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ============================================
    # 步骤 2: 检索知识（第一次，应该为空）
    # ============================================
    print(f"\n{'='*80}")
    print("步骤 2: 检索知识（第一次）")
    print(f"{'='*80}")
    
    task_query = "计算矩阵的特征值和特征向量，特征值需要按实部降序排序"
    
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
        result = kb.learn_from_trajectory(
            initial_program_path=initial_program_file,
            best_program_path=best_program_file,
            original_task=task_query,
            evaluator_file=evaluator_file,  # 传入 evaluator 文件路径
            metrics=None  # 让系统自动评估 best_program
        )
        
        print(f"\n学习结果:")
        print(f"  {'-'*70}")
        print(f"  提取的原则:")
        print(f"    {result['extracted_principle']}")
        print(f"\n  反事实测试代码长度: {len(result['counterfactual_test'])} 字符")
        print(f"  验证结果: {result['verification_outcome']}")
        print(f"  是否保存: {result['saved']}")
        print(f"  {'-'*70}")
        
        if result['saved']:
            print("✅ 原则已验证并保存到知识库")
        else:
            print("⚠️  原则验证失败，未保存到知识库")
        
        print("✅ 学习流程完成")
        
    except Exception as e:
        print(f"❌ 学习流程失败: {e}")
        import traceback
        print(f"\n详细错误信息:")
        traceback.print_exc()
        return False
    
    # ============================================
    # 步骤 4: 再次检索知识（应该能找到刚学习的知识）
    # ============================================
    print(f"\n{'='*80}")
    print("步骤 4: 再次检索知识（应该能找到刚学习的知识）")
    print(f"{'='*80}")
    
    try:
        retrieved2 = kb.retrieve_knowledge(task_query, k=3)
        print(f"\n查询: {task_query}")
        print(f"\n检索结果:")
        print(f"  - 原则数量: {len(retrieved2['principles'])}")
        print(f"  - 轨迹数量: {len(retrieved2['trajectories'])}")
        
        if retrieved2['principles']:
            print(f"\n找到的原则:")
            for i, principle in enumerate(retrieved2['principles'], 1):
                print(f"  {i}. {principle}")
        else:
            print("  ⚠️  没有找到原则（可能验证失败或检索问题）")
        
        if retrieved2['trajectories']:
            print(f"\n找到的轨迹:")
            for i, trajectory in enumerate(retrieved2['trajectories'], 1):
                print(f"  {i}. {trajectory[:200]}...")
        else:
            print("  ⚠️  没有找到轨迹（可能验证失败或检索问题）")
        
        # 验证是否找到了新学习的知识
        if result['saved']:
            if retrieved2['principles'] or retrieved2['trajectories']:
                print("\n✅ 成功检索到新学习的知识！")
            else:
                print("\n⚠️  虽然原则已保存，但检索时未找到（可能是向量检索的问题）")
        else:
            print("\n⚠️  原则未保存，所以检索不到是正常的")
        
        print("✅ 检索完成")
    except Exception as e:
        print(f"❌ 检索失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ============================================
    # 步骤 5: 测试不同的查询（语义相似）
    # ============================================
    print(f"\n{'='*80}")
    print("步骤 5: 测试不同的查询（语义相似）")
    print(f"{'='*80}")
    
    similar_queries = [
        "如何计算矩阵的特征向量",
        "矩阵特征值分解",
        "eigenvalue eigenvector computation"
    ]
    
    for query in similar_queries:
        try:
            retrieved = kb.retrieve_knowledge(query, k=2)
            print(f"\n查询: {query}")
            print(f"  找到 {len(retrieved['principles'])} 个原则, {len(retrieved['trajectories'])} 个轨迹")
        except Exception as e:
            print(f"❌ 查询 '{query}' 失败: {e}")
    
    print("\n✅ 所有测试完成！")
    return True

if __name__ == "__main__":
    success = test_full_workflow()
    sys.exit(0 if success else 1)

