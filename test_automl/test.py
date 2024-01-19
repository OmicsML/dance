import inspect
import sys


def function_one():
    pass


def function_two():
    pass


# 获取当前文件中的所有函数
current_functions = [(name, obj) for name, obj in inspect.getmembers(sys.modules[__name__]) if inspect.isfunction(obj)]

# 打印所有函数名
print("当前文件中的所有函数：", current_functions)
