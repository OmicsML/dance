import argparse
from pprint import pprint

from dance.datasets.spatial import CellTypeDeconvoDatasetLite
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
dataset = CellTypeDeconvoDatasetLite(data_id=args.dataset, data_dir=args.datadir)
sc_count = dataset.data["ref_sc_count"]
sc_meta = dataset.data["ref_sc_annot"]
spatial_count = dataset.data["mix_count"]
true_p = dataset.data["true_p"]
spatial_location = None if args.location_free else dataset.data["spatial_location"]
ct_select = sorted(set(sc_meta.cellType.unique().tolist()) & set(true_p.columns.tolist()))
print(f"{ct_select=}")

# Initialize and train moel
crd = Card(
    sc_count=sc_count,
    sc_meta=sc_meta,
    spatial_count=spatial_count,
    spatial_location=spatial_location,
    ct_varname="cellType",
    ct_select=ct_select,
    cell_varname=None,
    sample_varname=None,
)
crd.fit(max_iter=args.max_iter, epsilon=args.epsilon)

# Evaluate
pred = crd.predict()
mse = crd.score(pred, true_p[ct_select].values)

print(f"Predicted cell-type proportions of  sample 1: {pred[0].round(3)}")
print(f"True cell-type proportions of  sample 1: {true_p[ct_select].iloc[0].tolist()}")
print(f"mse = {mse:7.4f}")
"""To reproduce CARD benchmarks, please refer to command lines belows:

CARD synthetic
$ python card.py --dataset CARD_synthetic

GSE174746
$ python card.py --dataset GSE174746 --location_free

SPOTLight synthetic
$ python card.py --dataset SPOTLight_synthetic --location_free

"""
