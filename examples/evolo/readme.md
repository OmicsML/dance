做一些关于图和算法的分析

https://ai.google.dev/gemini-api/docs/rate-limits?hl=zh-cn
升一下级方便做实验，支付有困难的话就去淘宝

sk-acad184f8a7848918d7c9046affd1b65

先为每个算法找到一个在大多数数据集上合适的预处理流程，然后筛选出合适的算法，为这些算法在每个数据集上都找到合适的预处理流程。
https://aistudio.google.com/prompts/1zKDBdCj49JGD4jDzgzNlCWqDzdXZUGiL

python ../openevolve-run.py initial_program.py ../evaluator.py --config config.yaml

聚类，尤其是空间域识别部分，可以使用内部指标去优化预处理函数
外部指标的话，就说是元学习
内外部指标同时写上去，然后只用内部指标评估即可（聚类和空间域识别等任务）

提示词里面应该描述一下算法的(除stagate以外均需要修改)

修改聚类内部指标，使其可以描述空间和特征距离

spagcn以及louvain的指标已经修改

scdeepsort的修改词又改了，看看改后怎么样，可以的话就都移走吧
// "Human_Brain": \[
//     "--species",
//     "human",
//     "--tissue",
//     "Brain",
//     "--train_dataset",
//     "328",
//     "--test_dataset",
//     "138",
//     "--n_epochs",
//     "300",
//     "--num_runs",
//     "5",
//     "--device",
//     "cuda:0",
//     "--dense_dim",
//     "200"
// \],

brain暂时删掉了
使用open-evolve的可视化

#改成规则，而不是参考代码
