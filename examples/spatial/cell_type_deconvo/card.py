import argparse
from pprint import pprint

from dance.datasets.spatial import CellTypeDeconvoDataset
from dance.modules.spatial.cell_type_deconvo.card import Card

# TODO: make this a property of the dataset class?
DATASETS = ["CARD_synthetic", "GSE174746", "SPOTLight_synthetic"]

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataset", default="CARD_synthetic", choices=DATASETS, help="Name of the benchmarking dataset.")
parser.add_argument("--datadir", default="data/spatial", help="Directory to save the data.")
parser.add_argument("--max_iter", type=int, default=10, help="Maximum optimization iteration.")
parser.add_argument("--epsilon", type=float, default=1e-10, help="Optimization threshold.")
parser.add_argument("--location_free", action="store_true", help="Do not supply spatial location if set.")
args = parser.parse_args()
pprint(vars(args))

# Load dataset
dataset = CellTypeDeconvoDataset(data_dir=args.datadir, data_id=args.dataset)
data = dataset.load_data()

data.set_config(feature_channel=[None, "spatial"], feature_channel_type=["X", "obsm"],
                label_channel="cell_type_portion")
(x_count, x_spatial), y = data.get_data(split_name="test", return_type="numpy")
cell_types = data.data.obsm["cell_type_portion"].columns.tolist()

ref_adata = data.get_split_data("ref")
ref_count = ref_adata.to_df()
ref_annot = ref_adata.obs

model = Card(ref_count, ref_annot, ct_varname="cellType", ct_select=cell_types)
pred = model.fit_and_predict(x_count, x_spatial, max_iter=args.max_iter, epsilon=args.epsilon,
                             location_free=args.location_free)
mse = model.score(pred, y)

print(f"Predicted cell-type proportions of  sample 1: {pred[0].round(3)}")
print(f"True cell-type proportions of  sample 1: {y[0].round(3)}")
print(f"mse = {mse:7.4f}")
"""To reproduce CARD benchmarks, please refer to command lines belows:

CARD synthetic
$ python card.py --dataset CARD_synthetic

GSE174746
$ python card.py --dataset GSE174746 --location_free

SPOTLight synthetic
$ python card.py --dataset SPOTLight_synthetic --location_free

"""
