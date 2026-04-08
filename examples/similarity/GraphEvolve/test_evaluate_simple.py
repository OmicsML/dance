#!/usr/bin/env python3
"""测试 LamarckianKnowledgeBase 的完整工作流程 (Client-Server 版)

基于 eigenvectors_complex 例子测试：
1. 初始化知识库 (连接远程 Chroma Server)
2. 清理旧数据 (防止重复)
3. 检索知识（第一次应该为空）
4. 学习轨迹（learn_from_trajectory）
5. 再次检索知识（应该能检索到刚学习的知识）

"""

import asyncio
import os
import shutil
import sys

# 添加项目路径到 sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from lamarckian_knowledge_base import LamarckianKnowledgeBase

# ================= 配置区域 =================
# Chroma Server 配置
SERVER_HOST = "211.87.232.112"
SERVER_PORT = 8000

# 任务配置
task_query = os.environ.get("TASK_QUERY", "eigenvectors_complex")  # 提供默认值防止报错

# 路径配置
example_dir = "/mnt/nfs/zyxing/msu/dance_temp/dance/examples/evolo"
# 注意：确保 task_query 对应的目录存在，否则下方路径会出错
target_task_dir = os.path.join(example_dir, task_query)

evaluator_file = os.path.join(example_dir, "evaluator.py")
initial_program_file = os.path.join(target_task_dir, "initial_program.py")
best_program_file = os.path.join(target_task_dir, "openevolve_output", "best", "best_program.py")
config_yaml_path = os.path.join(target_task_dir, 'config.yaml')

# ================= 初始化知识库 =================
print(f"\n{'='*80}")
print("初始化知识库 (Connecting to Chroma Server)")
print(f"{'='*80}")
print(f"Server: http://{SERVER_HOST}:{SERVER_PORT}")

try:
    # 🌟 修改点：不再传递 path，而是传递 host 和 port
    kb = LamarckianKnowledgeBase(host=SERVER_HOST, port=SERVER_PORT)
    print("✅ 知识库连接成功")
except Exception as e:
    print(f"❌ 知识库连接失败: {e}")
    print("请检查：\n1. 服务器 211.87.232.112 是否已启动 chroma run\n2. 端口 8000 是否开放")
    sys.exit(1)


def test_full_workflow():
    """测试完整的工作流程."""

    print("=" * 80)
    print("测试 LamarckianKnowledgeBase 完整工作流程")
    print("=" * 80)
    print(f"Task Query: {task_query}")

    # 检查 API key
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("\n⚠️  警告: 未设置 DASHSCOPE_API_KEY 环境变量")
        print("   测试将在 LLM 调用环节失败")

    # ============================================
    print(f"\n{'='*80}")
    print("步骤 0: 清理与当前 Task 相关的旧数据")
    print(f"{'='*80}")

    try:
        # 获取所有数据
        all_memories = kb.list_all_memories()
        principles = all_memories.get("principles", [])
        trajectories = all_memories.get("trajectories", [])

        ids_to_delete = []

        # 策略：只要 metadata 中的 source_task 与当前 task_query 相同，就删除
        # 这比先检索再删除更彻底，能保证测试环境纯净
        for p in principles:
            if p.get("metadata", {}).get("source_task") == task_query:
                ids_to_delete.append(p["id"])

        for t in trajectories:
            if t.get("metadata", {}).get("source_task") == task_query:
                ids_to_delete.append(t["id"])

        if ids_to_delete:
            print(f"发现 {len(ids_to_delete)} 条旧数据，正在删除...")
            # 直接调用底层 collection 删除
            kb._chroma_collection.delete(ids=ids_to_delete)
            print(f"✅ 已删除 {len(ids_to_delete)} 条旧数据")
        else:
            print("没有发现旧数据，环境干净")

    except Exception as e:
        print(f"⚠️  清理数据时出现警告: {e}")

    # ============================================
    # 步骤 2: 检索知识（第一次，应该为空）
    # ============================================
    print(f"\n{'='*80}")
    print("步骤 2: 检索知识（第一次，预期为空）")
    print(f"{'='*80}")

    try:
        retrieved = kb.retrieve_knowledge(task_query, k=3)
        print(f"检索结果: {len(retrieved['principles'])} 原则, {len(retrieved['trajectories'])} 轨迹")

        if len(retrieved['principles']) == 0 and len(retrieved['trajectories']) == 0:
            print("✅ 符合预期：知识库为空")
        else:
            print("⚠️  注意：知识库中仍有相关数据（可能是其他 Task 的相似内容）")

    except Exception as e:
        print(f"❌ 检索失败: {e}")
        return False

    # ============================================
    # 步骤 3: 学习轨迹（learn_from_trajectory）
    # ============================================
    print(f"\n{'='*80}")
    print("步骤 3: 学习轨迹（learn_from_trajectory）")
    print(f"{'='*80}")
    print(f"Initial: {initial_program_file}")
    print(f"Best:    {best_program_file}")

    # 检查文件
    if not os.path.exists(initial_program_file):
        print(f"❌ 找不到初始程序: {initial_program_file}")
        return False
    if not os.path.exists(best_program_file):
        print(f"❌ 找不到最优程序: {best_program_file}")
        return False

    try:
        result = asyncio.run(
            kb.learn_from_trajectory(
                initial_program_path=initial_program_file,
                best_program_path=best_program_file,
                original_task=task_query,
                evaluator_file=evaluator_file,
                metrics=None,  # 自动评估
                config=config_yaml_path,
            ))

        print(f"\n学习结果摘要:")
        print(f"  - 提取原则总数: {len(result['all_principles'])}")
        print(f"  - 验证并通过数: {result['saved_count']}")

        if result['saved_count'] > 0:
            print("✅ 学习成功，数据已写入远程数据库")
        else:
            print("⚠️  学习完成但未保存任何原则（可能是反事实验证未通过）")

    except Exception as e:
        print(f"❌ 学习流程出错: {e}")
        import traceback
        traceback.print_exc()
        return False

    # ============================================
    # 步骤 4: 验证检索（验证是否真的存进去了）
    # ============================================
    print(f"\n{'='*80}")
    print("步骤 4: 验证检索 (确认数据已入库)")
    print(f"{'='*80}")

    # 等待一秒让索引刷新
    import time
    time.sleep(1)

    similar_queries = [task_query]

    for query in similar_queries:
        try:
            retrieved = kb.retrieve_knowledge(query, k=2)
            print(f"查询: '{query}'")
            print(f"  -> 找到 {len(retrieved['principles'])} 原则")

            if retrieved['principles']:
                print(f"  -> 内容示例: {retrieved['principles'][0][:60]}...")
                print("✅ 验证成功：能检索到新学习的知识")
            else:
                if result['saved_count'] > 0:
                    print("❌ 验证失败：已保存但无法检索（可能是 Embedding 维度问题或索引延迟）")
                    return False
                else:
                    print("⚠️  验证跳过：之前没有保存任何原则")

        except Exception as e:
            print(f"❌ 查询失败: {e}")
            return False

    print("\n🎉 所有流程测试通过！")
    return True


if __name__ == "__main__":
    success = test_full_workflow()
    sys.exit(0 if success else 1)
