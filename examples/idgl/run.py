import json
import os
import subprocess
import sys
from dance.settings import EXAMPLESDIR

method_name=os.environ['idgl_method_name']
# 1. 确定路径
config_path = os.path.join(EXAMPLESDIR, "evolo/benchmarks_config.json")
# 假设 main.py 在 EXAMPLESDIR/idgl 目录下，如果不是请修改这里
work_dir = os.path.join(EXAMPLESDIR, f"idgl/{method_name}") 

method_name = f"{method_name.split('_')[1].lower()}_benchmarks"
# 检查配置文件是否存在
if not os.path.exists(config_path):
    print(f"错误: 找不到配置文件 {config_path}")
    sys.exit(1)

with open(config_path, "r") as f:
    benchmarks = json.load(f).get(method_name, {})

if not benchmarks:
    print(f"警告: {method_name} 配置为空或未找到！")

for dataset_name, args in benchmarks.items():
    command = [sys.executable, "main.py"] + args
    if method_name in ["scgat_benchmarks"]:
        command.append('--chunk_size')
        command.append('500')
    
    print("-" * 50)
    print(f"正在启动数据集: {dataset_name}")
    print(f"工作目录: {work_dir}")
    print(f"完整命令: {' '.join(command)}")
    print("-" * 50)

    try:
        # 【关键修改】
        # 1. 去掉 capture_output=True，让输出直接显示在屏幕上
        # 2. 加上 cwd=work_dir，确保能找到 main.py
        subprocess.run(command, check=True, cwd=work_dir)
        
        print(f"\n>>> 数据集 {dataset_name} 运行完成")
        
    except subprocess.CalledProcessError as e:
        print(f"\nXXX 运行出错 (数据集: {dataset_name})")
        print(f"退出代码: {e.returncode}")
        # 不需要打印 e.stderr，因为去掉 capture_output 后错误已经直接显示在屏幕上了
    except FileNotFoundError as e:
        print(e)
        print(f"\nXXX 错误: 在 {work_dir} 下找不到 main.py，请检查路径。")