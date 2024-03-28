import re
from pathlib import Path

import pandas as pd


def list_files(directory, file_name="out.log", save_path="summary_file.csv"):
    ans = []
    path = Path(directory)
    for file_path in path.rglob('*'):
        if file_path.is_file():
            if file_path.name == file_name:
                with open(file_path) as f:
                    lines = [str(line) for line in f.readlines()]
                    pattern = r'(?<=^wandb: 🧹 View sweep at ).*'
                    matches = [re.search(pattern, string).group() for string in lines if re.search(pattern, string)]
                    counts = {x: matches.count(x) for x in matches}
                    step2 = max(counts, key=counts.get)
                    step3_list = [key for key in counts if key != step2]
                algorithm, dataset = file_path.relative_to(directory).parts[:2]
                try:
                    with open(Path(file_path.parent, "results/pipeline/pipeline_summary_pattern.txt")) as file:
                        summary_pattern = file.read()
                except FileNotFoundError:
                    summary_pattern = None
                ans.append({
                    "method": algorithm,
                    "dataset": dataset,
                    "step2": step2,
                    "step3_list": str(step3_list),
                    "step2_summary": summary_pattern,
                    "step3_summary": None
                })
    pd.DataFrame(ans).to_csv(save_path)


if __name__ == "__main__":
    list_files("/home/zyxing/dance/examples/tuning")
