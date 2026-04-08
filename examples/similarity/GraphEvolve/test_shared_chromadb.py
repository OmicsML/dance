#!/usr/bin/env python3
"""测试多服务器间共享 ChromaDB (Client-Server 模式专用)

使用前请确保：
1. 服务器 A (211.87.232.112) 已启动 Chroma Server
2. 如果之前有 384 维的旧数据，建议在服务器上先删除旧集合或清空数据

启动 Server 命令：
    chroma run --host 0.0.0.0 --port 8000 --path /mnt/nfs/zyxing/msu/dance_temp/dance/chroma_server_data

"""

import sys
import time

from lamarckian_knowledge_base import LamarckianKnowledgeBase
from langchain_core.documents import Document

# 配置
SERVER_HOST = "211.87.232.112"
SERVER_PORT = 8000

# 测试用的 task query
TEST_TASK = "test_shared_database_v2"


def test_connection():
    """测试 1: 连接到 Chroma Server."""
    print("\n" + "=" * 60)
    print("测试 1: 连接到 Chroma Server")
    print("=" * 60)

    try:
        # 实例化：直接指定 Host 和 Port
        kb = LamarckianKnowledgeBase(host=SERVER_HOST, port=SERVER_PORT)
        print(f"✅ 成功连接到 Chroma Server: http://{SERVER_HOST}:{SERVER_PORT}")

        # 检查当前知识库中的数据量
        all_memories = kb.list_all_memories()

        # 检查是否有错误
        if isinstance(all_memories, dict) and "error" in all_memories:
            print(f"❌ 读取列表失败: {all_memories['error']}")
            return None

        p_count = len(all_memories.get('principles', []))
        t_count = len(all_memories.get('trajectories', []))

        print(f"   当前知识库状态:")
        print(f"   - 原则数量: {p_count}")
        print(f"   - 轨迹数量: {t_count}")

        return kb

    except Exception as e:
        print(f"❌ 连接失败: {e}")
        print("\n请确认 Chroma Server 已启动且防火墙已放行端口 8000")
        return None


def test_write_and_read(kb):
    """测试 2: 写入并读取数据 (重点验证维度兼容性)"""
    print("\n" + "=" * 60)
    print("测试 2: 写入并读取数据 (1024维向量测试)")
    print("=" * 60)

    # 1. 清理旧数据
    print("\n🧹 清理旧的测试数据...")
    try:
        all_memories = kb.list_all_memories()
        ids_to_delete = []

        for p in all_memories.get("principles", []):
            if p.get("metadata", {}).get("source_task") == TEST_TASK:
                ids_to_delete.append(p["id"])

        for t in all_memories.get("trajectories", []):
            if t.get("metadata", {}).get("source_task") == TEST_TASK:
                ids_to_delete.append(t["id"])

        if ids_to_delete:
            kb._chroma_collection.delete(ids=ids_to_delete)
            print(f"   已清理 {len(ids_to_delete)} 条旧数据")
        else:
            print("   无旧数据需清理")

    except Exception as e:
        print(f"   ⚠️ 清理时警告: {e}")

    # 2. 写入测试数据
    print("\n📝 写入测试数据...")
    print("   (正在调用 DashScopeEmbeddings 生成 1024 维向量，请稍候...)")

    test_principle = "测试原则: Client-Server模式必须确保Embedding维度一致(1024)"
    test_trajectory = "测试轨迹: 这是一个用于验证远程写入的测试记录"

    try:
        # 重要：使用 vector_store.add_documents
        # 这会自动调用类中定义的 self.embeddings (1024维) 来生成向量
        # 从而避免 "expecting 384 got 1024" 的错误

        docs = [
            Document(page_content=test_principle, metadata={
                "source_task": TEST_TASK,
                "type": "principle",
                "verified": "true"
            }),
            Document(page_content=test_trajectory, metadata={
                "source_task": TEST_TASK,
                "type": "trajectory"
            })
        ]

        # 写入
        kb.vector_store.add_documents(docs)
        print("   ✅ 写入操作完成")

        # 稍等一下让索引生效
        time.sleep(1)

    except Exception as e:
        print(f"   ❌ 写入失败: {e}")
        if "dimension" in str(e).lower():
            print("\n   🔴 严重错误：维度不匹配！")
            print("   请登录服务器删除旧的 Collection，或者在代码中更改集合名称。")
        return False

    # 3. 检索测试
    print("\n🔍 检索测试数据...")
    try:
        retrieved = kb.retrieve_knowledge(TEST_TASK, k=3)
        print(f"   查询 '{TEST_TASK}':")
        print(f"   - 找到 {len(retrieved['principles'])} 个原则")
        print(f"   - 找到 {len(retrieved['trajectories'])} 个轨迹")

        if retrieved['principles']:
            print(f"\n   原则内容示例: {retrieved['principles'][0]}")

        success = len(retrieved['principles']) > 0
        print(f"\n   {'✅ 检索成功' if success else '❌ 检索失败'}")
        return success

    except Exception as e:
        print(f"   ❌ 检索失败: {e}")
        return False


def test_shared_access(kb):
    """测试 3: 模拟从另一台机器连接."""
    print("\n" + "=" * 60)
    print("测试 3: 验证数据共享 (模拟第二台客户端)")
    print("=" * 60)

    print("\n🔄 建立第二个连接实例...")
    try:
        # 模拟第二个客户端
        kb2 = LamarckianKnowledgeBase(host=SERVER_HOST, port=SERVER_PORT)

        retrieved = kb2.retrieve_knowledge(TEST_TASK, k=3)
        print(f"   从新连接中检索到:")
        print(f"   - {len(retrieved['principles'])} 个原则")

        if retrieved['principles']:
            print(f"   内容验证: '{retrieved['principles'][0][:20]}...'")

        success = len(retrieved['principles']) > 0
        return success

    except Exception as e:
        print(f"   ❌ 验证失败: {e}")
        return False


def cleanup(kb):
    """最后清理."""
    print("\n" + "=" * 60)
    print("清理本次测试产生的数据")
    print("=" * 60)
    try:
        # 获取所有 ID 并删除本次测试的 task
        data = kb.list_all_memories()
        ids_to_del = []
        count = 0

        # 简单的过滤逻辑
        all_items = data.get("principles", []) + data.get("trajectories", [])
        for item in all_items:
            if item.get("metadata", {}).get("source_task") == TEST_TASK:
                ids_to_del.append(item["id"])

        if ids_to_del:
            kb._chroma_collection.delete(ids=ids_to_del)
            print(f"   ✅ 成功删除了 {len(ids_to_del)} 条测试残留数据")
        else:
            print("   没有发现残留数据")

    except Exception as e:
        print(f"   ⚠️ 清理出错: {e}")


def main():
    print("=" * 60)
    print("ChromaDB Client-Server 最终集成测试")
    print("=" * 60)
    print(f"Target Server: {SERVER_HOST}:{SERVER_PORT}")

    # 1. 连接
    kb = test_connection()
    if kb is None:
        sys.exit(1)

    # 2. 写入与读取 (含 Embedding 生成)
    write_ok = test_write_and_read(kb)

    # 3. 共享验证
    share_ok = False
    if write_ok:
        share_ok = test_shared_access(kb)

    # 4. 清理
    cleanup(kb)

    # 总结
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    print(f"   连接测试: {'✅ 通过' if kb else '❌ 失败'}")
    print(f"   写/读测试: {'✅ 通过' if write_ok else '❌ 失败'}")
    print(f"   共享测试: {'✅ 通过' if share_ok else '❌ 失败'}")

    if write_ok and share_ok:
        print("\n🎉 系统运行完美！Client-Server 模式及 1024 维向量配置正确。")
        sys.exit(0)
    else:
        print("\n⚠️ 测试未完全通过，请检查日志。")
        sys.exit(1)


if __name__ == "__main__":
    main()
