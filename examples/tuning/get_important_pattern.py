import itertools
import pathlib
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikit_posthocs as sp
import seaborn as sns
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules
from networkx import parse_adjlist
from scipy import stats

metric_name = "acc"
ascending = False


def get_important_pattern(test_accs, vis=True, alpha=0.8, title="",test_acc_names=None):
    medians = [np.median(group) for group in test_accs]
    _, p_value = stats.kruskal(*test_accs)
    if vis:
        fig = plt.figure(figsize=(12, 4))
        sns.boxplot(data=test_accs)
        plt.xticks(list(range(len(test_accs))), ([f"{i}" for i in range(len(test_accs))] if  test_acc_names is None else test_acc_names),rotation=45, fontsize=10)
        plt.title(title)
        plt.show()
    if p_value < alpha:
        data = test_accs
        p_values_matrix = sp.posthoc_dunn(a=data)
        sorted_indices = np.argsort(np.argsort(medians * -1 if ascending else medians))
        ranks = {
            index: {
                "rank": rank,
                "before": None,
                "after": [],
                "real_rank": rank
            }
            for index, rank in enumerate(sorted_indices)
        }
        for (rank1, rank2) in combinations(range(max(sorted_indices) + 1), 2):
            for idx1 in [index for index, value in ranks.items() if value["rank"] == rank1]:
                for idx2 in [index for index, value in ranks.items() if value["rank"] == rank2]:
                    if p_values_matrix.iloc[idx1, idx2] > alpha:
                        if ranks[idx2]["before"] is None:
                            ranks[idx1]["after"].append(idx2)
                            ranks[idx2]["before"] = idx1

        def change_real_rank(rank_item, real_rank):
            rank_item["real_rank"] = real_rank
            for idx in rank_item["after"]:
                change_real_rank(ranks[idx], real_rank)

        for rank_item in ranks.values():
            if rank_item["before"] is None:
                for idx in rank_item["after"]:
                    change_real_rank(ranks[idx], rank_item["real_rank"])
        return [v["real_rank"] for k, v in ranks.items()]
    else:
        if vis:
            print("No significant differences found between the groups.")
        return []


def get_com(step2_data, r=2, alpha=0.8, columns=None, vis=True):
    ans = []
    for com in itertools.combinations(columns, r):
        test_accs_arrays = []
        for g in step2_data.groupby(by=list(com)):
            test_accs_arrays.append({"name": g[0], metric_name: list(g[1][metric_name])})
        test_accs = [i[metric_name] for i in test_accs_arrays]
        test_acc_names = [i["name"] for i in test_accs_arrays]
        final_ranks = get_important_pattern(test_accs, alpha=alpha, title=" ".join(list(com)), vis=vis,test_acc_names=[" ".join(test_acc_name) for test_acc_name in test_acc_names])
        if len(final_ranks) > 0:
            max_rank = max(final_ranks)
            max_rank_count = final_ranks.count(max_rank)
            if max_rank_count < len(final_ranks) / 2:
                for index, (test_acc_name, rank) in enumerate(zip(test_acc_names, final_ranks)):
                    if rank == max_rank:
                        if vis:
                            print(f"index={index},name={test_acc_name},rank={rank}")
                        ans.append(test_acc_name if isinstance(test_acc_name, tuple) else (test_acc_name, ))
    return ans
def draw_graph(rules, rules_to_show):
  import networkx as nx  
  G1 = nx.DiGraph()
   
  color_map=[]
  N = 50
  colors = np.random.rand(N)    
  strs=['R0', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'R11']   
   
   
  for i in range (rules_to_show):      
    G1.add_nodes_from(["R"+str(i)])
    
     
    for a in rules.iloc[i]['antecedents']:
                
        G1.add_nodes_from([a])
        
        G1.add_edge(a, "R"+str(i), color=colors[i] , weight = 2)
       
    for c in rules.iloc[i]['consequents']:
             
            G1.add_nodes_from([c])
            
            G1.add_edge("R"+str(i), c, color=colors[i],  weight=2)
 
  for node in G1:
       found_a_string = False
       for item in strs: 
           if node==item:
                found_a_string = True
       if found_a_string:
            color_map.append('yellow')
       else:
            color_map.append('green')       
 
 
   
  edges = G1.edges()
  colors = [G1[u][v]['color'] for u,v in edges]
  weights = [G1[u][v]['weight'] for u,v in edges]
 
  pos = nx.spring_layout(G1, k=16, scale=1)
  nx.draw(G1, pos, node_color = color_map, edge_color=colors, width=weights, font_size=16, with_labels=False)            
   
  for p in pos:  # raise text positions
           pos[p][1] += 0.07
  nx.draw_networkx_labels(G1, pos)
  plt.show()

def get_frequent_itemsets(step2_data, threshold_per=0.1,vis=False):
    threshold = int(len(step2_data) * threshold_per)
    df_sorted = step2_data.sort_values(metric_name, ascending=ascending)
    top_10_percent = df_sorted.head(threshold)
    columns = sorted([col for col in step2_data.columns if col.startswith("pipeline")])
    transactions = top_10_percent[columns].values.tolist()
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    frequent_itemsets = apriori(df, min_support=0.3, use_colnames=True)
    # print(frequent_itemsets)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
    if vis:
        # print(frequent_itemsets)
        # print(frequent_itemsets)
        # draw_graph(rules=rules,rules_to_show=10)
        frequent_itemsets_copy=frequent_itemsets.copy()
        frequent_itemsets_copy=frequent_itemsets_copy.sort_values(by="support")
        frequent_itemsets_copy.plot(x="itemsets",y="support",kind="bar")
        plt.xticks(rotation=30, fontsize=7)
        # print(type(rules))
    return [tuple(a) for a in frequent_itemsets["itemsets"]]


def get_com_all(step2_data, vis=True, alpha=0.8):
    ans = []
    columns = sorted([col for col in step2_data.columns if col.startswith("pipeline")])
    for i in range(1, len(columns)):
        ans += get_com(step2_data, i, columns=columns, vis=vis, alpha=alpha)
    return ans


def summary_pattern(data_path, alpha=0.8, vis=False):
    step2_origin_data = pd.read_csv(data_path)
    step2_data = step2_origin_data.dropna()
    com_ans = get_com_all(step2_data, vis=vis, alpha=alpha)
    apr_ans = get_frequent_itemsets(step2_data,vis=vis)
    return list(set(com_ans) & set(apr_ans))


# def list_files(directory,file_name="best_test_acc.csv",save_path="summary_file"):
#     ans=[]
#     path = Path(directory)
#     for file_path in path.rglob('*'):
#         if file_path.is_file():
#             if file_path.name==file_name:
#                 algorithm,dataset=file_path.relative_to(directory).parts[:2]
#                 ans.append({"algorithm":algorithm,"dataset":dataset,"summary_pattern":summary_pattern(file_path)})
#     pd.DataFrame(ans).to_csv(save_path)
def list_files(directories, file_name="best_test_acc.csv", alpha=0.8, vis=False):
    for directory in directories:
        path = Path(directory)
        for file_path in path.rglob('*'):
            if file_path.is_file():
                if file_path.name == file_name:
                    print(file_path)
                    with open(Path(file_path.parent.resolve(), "pipeline_summary_pattern.txt"), 'w') as f:
                        f.write(str(summary_pattern(file_path, alpha=alpha, vis=vis)))


if __name__ == "__main__":
    # directories = []
    # for path in Path('/home/zyxing/dance/examples/tuning').iterdir():
    #     if path.is_dir():
    #         if str(path.name).startswith("cluster"):
    #             directories.append(path)
    # list_files(directories)

    print(summary_pattern("/home/zyxing/dance/examples/tuning/cluster_graphsc/mouse_ES_cell/results/pipeline/best_test_acc.csv",alpha=0.3,vis=False))
