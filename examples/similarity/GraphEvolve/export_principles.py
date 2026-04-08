#!/usr/bin/env python3
"""导出知识库中所有 principles 的 content 和 source_task 到 txt 文件."""
import os
import sys

# 添加当前目录到 Python 路径，以便导入 lamarckian_knowledge_base
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lamarckian_knowledge_base import LamarckianKnowledgeBase

# ================= 配置区域 =================
SERVER_HOST = "211.87.232.112"
SERVER_PORT = 8000
OUTPUT_FILE = "principles_export.txt"


# ================= 主程序 =================
def export_principles():
    print(f"\n{'='*80}")
    print("导出知识库中的 Principles")
    print(f"{'='*80}")
    print(f"Server: http://{SERVER_HOST}:{SERVER_PORT}")
    print(f"输出文件: {OUTPUT_FILE}")

    # 1. 连接知识库
    try:
        kb = LamarckianKnowledgeBase(host=SERVER_HOST, port=SERVER_PORT)
        print("✅ 知识库连接成功")
    except Exception as e:
        print(f"❌ 知识库连接失败: {e}")
        sys.exit(1)

    # 2. 获取所有 memories
    try:
        memories = kb.list_all_memories()
        principles = memories.get('principles', [])
        print(f"\n📊 共找到 {len(principles)} 条 principles")
    except Exception as e:
        print(f"❌ 获取知识库数据失败: {e}")
        sys.exit(1)

    # 3. 导出到 txt 文件
    if not principles:
        print("\n⚠️  没有找到任何 principles")
        # 创建空文件
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write("# 导出时间: " + str(__import__('datetime').datetime.now()) + "\n")
            f.write("# 原则数量: 0\n")
            f.write("\n# 没有找到任何 principles\n")
        print(f"已创建空文件: {OUTPUT_FILE}")
        return

    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            # 写入文件头
            f.write("=" * 80 + "\n")
            f.write("Principles 导出文件\n")
            f.write("=" * 80 + "\n")
            f.write(f"导出时间: {__import__('datetime').datetime.now()}\n")
            f.write(f"原则总数: {len(principles)}\n")
            f.write("=" * 80 + "\n\n")

            # 写入每一条 principle
            for i, principle in enumerate(principles, 1):
                content = principle.get('content', '')
                metadata = principle.get('metadata', {})
                source_task = metadata.get('source_task', 'Unknown') if isinstance(metadata, dict) else 'Unknown'
                principle_id = principle.get('id', f'ID_{i}')

                # 写入分隔符
                f.write("-" * 80 + "\n")
                f.write(f"[{i}] Principle ID: {principle_id}\n")
                f.write("-" * 80 + "\n")

                # 写入 source_task
                f.write(f"Source Task: {source_task}\n")

                # 写入 content
                f.write(f"\nContent:\n{content}\n")

                # 写入元数据（可选，如果还有其他有用的字段）
                if isinstance(metadata, dict):
                    other_keys = [k for k in metadata.keys() if k != 'source_task']
                    if other_keys:
                        f.write(f"\nOther Metadata:\n")
                        for key in other_keys:
                            f.write(f"  {key}: {metadata[key]}\n")

                f.write("\n")  # 空行分隔

            # 写入文件尾
            f.write("=" * 80 + "\n")
            f.write("导出完成\n")
            f.write("=" * 80 + "\n")

        print(f"✅ 成功导出 {len(principles)} 条 principles 到: {OUTPUT_FILE}")

    except Exception as e:
        print(f"❌ 导出失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    export_principles()
