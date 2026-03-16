from datasets import Dataset, Features, Value

# 1. 定义数据结构 (Schema)
# 这一步非常重要！强制规定 'method' 和 'code' 必须是 string 类型
# 这样以后就算存入空值，也不会变成 float 导致报错
my_features = Features({
    'method': Value('string'),
    'code': Value('string')
})

# 2. 准备初始数据
# 建议至少放一条非空数据，或者放空列表也可以，只要 features 定义了就行
# 这里我们放一条测试数据，确保一切正常
initial_data = [
    {
        'method': 'init_setup',
        'code': '# This is the initial setup code.'
    }
]

# 3. 创建 Dataset
dataset = Dataset.from_list(initial_data, features=my_features)

# 4. 推送到 Hugging Face
# 注意：这会自动在你的账户下创建仓库
repo_id = "zhongyuxing/Graph_Structure_Learning_Pseudocode_new_new"
dataset.push_to_hub(repo_id, split="train")

print(f"成功创建新数据集：{repo_id}")