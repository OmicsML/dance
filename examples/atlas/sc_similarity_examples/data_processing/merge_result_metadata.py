tissues = ["blood", "brain", "heart", "intestine", "kidney", "lung", "pancreas"]
import pandas as pd

from dance.settings import ATLASDIR, SIMILARITYDIR

if __name__ == "__main__":
    for tissue in tissues:
        metadata_df = pd.read_csv(ATLASDIR / f"metadatas/{tissue}_metadata.csv")
        sweep_result_df = pd.read_csv(ATLASDIR / f"sweep_results/{tissue.capitalize()}_ans.csv")
        sweep_result_df = sweep_result_df.rename(columns={"Dataset_id": "dataset_id"})
        sweep_result_df["dataset_id"] = sweep_result_df["dataset_id"].str.split('(').str[0]
        result_df = metadata_df.merge(sweep_result_df, how="outer", on="dataset_id")
        #     result_df.to_csv(SIMILARITYDIR / f"data/results/{tissue}_result.csv")
        # for tissue in tissues:
        #     df=pd.read_csv(SIMILARITYDIR / f"data/results/{tissue}_result.csv")
        with pd.ExcelWriter(SIMILARITYDIR / "data/Cell Type Annotation Atlas.xlsx", mode='a',
                            if_sheet_exists='replace') as writer:
            result_df.to_excel(writer, sheet_name=tissue)
