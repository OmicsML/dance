import argparse
from pprint import pprint

from dance.datasets.spatial import CellTypeDeconvoDataset
from dance.modules.spatial.cell_type_deconvo.card import Card
from dance.transforms import (CellTopicProfile, FilterGenesCommon, FilterGenesMarker, FilterGenesMatch,
                              FilterGenesPercentile)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataset", default="CARD_synthetic", choices=CellTypeDeconvoDataset.DATASETS)
parser.add_argument("--datadir", default="data/spatial", help="Directory to save the data.")
parser.add_argument("--max_iter", type=int, default=10, help="Maximum optimization iteration.")
parser.add_argument("--epsilon", type=float, default=1e-10, help="Optimization threshold.")
parser.add_argument("--location_free", action="store_true", help="Do not supply spatial location if set.")
args = parser.parse_args()
pprint(vars(args))

# Load dataset
dataset = CellTypeDeconvoDataset(data_dir=args.datadir, data_id=args.dataset)
data = dataset.load_data()
cell_types = data.data.obsm["cell_type_portion"].columns.tolist()

CellTopicProfile(ct_select=cell_types, ct_key="cellType", batch_key=None, split_name="ref", method="mean",
                 log_level="INFO")(data)
FilterGenesMatch(prefixes=["mt-"], case_sensitive=False, log_level="INFO")(data)
FilterGenesCommon(split_keys=["ref", "test"], log_level="INFO")(data)
FilterGenesMarker(ct_profile_channel="CellTopicProfile", threshold=1.25, log_level="INFO")(data)
FilterGenesPercentile(min_val=1, max_val=99, mode="rv", log_level="INFO")(data)

data.set_config(feature_channel=[None, "spatial"], feature_channel_type=["X", "obsm"],
                label_channel="cell_type_portion")
(x_count, x_spatial), y = data.get_data(split_name="test", return_type="numpy")
# TODO: adapt card to use basis.T
# TODO: use "auto"/None option for ct_select
basis = data.get_feature(return_type="default", channel="CellTopicProfile", channel_type="varm").T

model = Card(basis)
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
