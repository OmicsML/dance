import argparse
import os

from dance.utils import try_import

entity = "xzy11632"
project = "dance-dev"
wandb = try_import("wandb")
parser = argparse.ArgumentParser()
parser.add_argument("--sweep_id", type=str, default="c3yy5fd3")
args = parser.parse_args()
sweep_id = args.sweep_id

import time


def delete_unfinished_runs(sweep_id, max_attempts=3, check_interval=1):
    """删除所有未完成的运行，并确保删除成功.

    参数:
    - sweep_id: sweep对象的id
    - max_attempts: 最大重试次数
    - check_interval: 每次检查间隔的秒数

    """
    attempt = 0
    while attempt < max_attempts:
        sweep = wandb.Api(timeout=1000).sweep(f"{entity}/{project}/{sweep_id}")
        # 检查是否还有未完成的运行
        unfinished_runs = [run for run in sweep.runs if run.state != 'finished']

        if not unfinished_runs:
            print("所有运行都已完成或已删除")
            break

        print(f"第 {attempt + 1} 次尝试删除 {len(unfinished_runs)} 个未完成的运行")

        # 尝试删除所有未完成的运行
        for run in unfinished_runs:
            try:
                run.delete()
            except Exception as e:
                print(f"删除运行 {run.id} 时出错: {str(e)}")

        # 等待一段时间后再次检查
        time.sleep(check_interval)
        attempt += 1

    # 最终检查
    remaining_unfinished = [run for run in sweep.runs if run.state != 'finished']
    if remaining_unfinished:
        print(f"警告: 仍有 {len(remaining_unfinished)} 个运行未能成功删除")
        return False
    return True


# 使用示例
success = delete_unfinished_runs(sweep_id)
if success:
    print("所有未完成的运行已成功删除")
else:
    print("部分运行可能未能成功删除")
os.system(f"wandb sweep --resume {entity}/{project}/{sweep_id}")
