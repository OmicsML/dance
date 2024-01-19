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

import sys


def set_method_name(func):

    def wrapper(*args, **kwargs):
        method_name = func.__name__ + "_"
        print(method_name)
        result = func(*args, **kwargs)
        return result

    return wrapper


@set_method_name
def function1():
    method_name = str(sys._getframe().f_code.co_name) + "_"
    print("函数 1 被调用")


function1()
