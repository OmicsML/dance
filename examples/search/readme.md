不要直接迁移，而是比较使用类似代码会不会好一些
将伪代码定义为prompt，参考https://github.com/algorithmicsuperintelligence/openevolve/tree/main/examples/llm_prompt_optimization
dataset里面可以包含不同方法，评估的时候过滤一下就可以，最好不同方法生成的伪代码可以放在同一个文件夹里面
伪代码应该是算法的伪代码

匹配的预处理以prompt的形式返回给原本的算法进行重新优化


1 origin   2 search  3 search+recommend   4 search+recommend(search)

可以拿wandb做参数搜索，简单比较一下。
没有必要使用autogl搜索，我们自己定制了函数的所有空间，可以尝试替换。