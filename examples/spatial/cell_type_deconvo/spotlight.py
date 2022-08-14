import sys
import os.path as o
#use if running from dance/examples/spatial/cell_type_deconvo
root_path=o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "../../.."))
sys.path.append(root_path)


from dance.datasets.spatial import CellTypeDeconvoDataset
from dance.datasets.spatial import CellTypeDeconvoDatasetLite
from dance.modules.spatial.cell_type_deconvo.spotlight import SPOTlight
import numpy as np
import torch
import torch.nn.functional as F
# Set random seed
seed = 123
np.random.seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# TODO: make this a property of the dataset class?
DATASETS = ["CARD_synthetic", "GSE174746", "SPOTLight_synthetic", "toy1", "toy2"]

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataset", default="toy2", choices=DATASETS, help="Name of the dataset.")
parser.add_argument("--datadir", default="data/spatial", help="Directory to save the data.")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
parser.add_argument("--rank", type=float, default=10, help="Rank of the NMF module.")
parser.add_argument("--bias", type=bool, default=False, help="Include/Exclude bias term.")
parser.add_argument("--max_iter", type=int, default=4000, help="Maximum optimization iteration.")
parser.add_argument("--epsilon", type=float, default=1e-10, help="Optimization threshold.")
args = parser.parse_args()
pprint(vars(args))


dataset = CellTypeDeconvoDatasetLite(data_id=args.dataset, data_dir=args.datadir)
X=torch.Tensor(dataset.data['ref_sc_count'].values)
Y=torch.Tensor(dataset.data['mix_count'].values)

#set cell-types 
cell_types =np.array([np.repeat(i+1, X.shape[1]//3) for i in range(3)]).flatten()


#run deconvolution with SPOTlight (NMFReg + additional NNLS layer) - set rank for NMF dim red.
decon_model = SPOTlight(in_dim=X.shape, hid_dim=np.unique(cell_types).shape[0], out_dim=Y.shape[1], rank=args.rank, device=device)

#fit using reference scRNA X, with cell_type labels
decon_model.fit(x=X,y=Y, cell_types=cell_types)

#decon_model.P gives fitted cell-type proportions for Y 

#predict proportions at each spot
prop_pred = decon_model.predict(x=X,y=Y,cell_types=cell_types)
prop_pred = prop_pred / torch.sum(prop_pred, 0)
#print for first spot
print('Estimated cell-type proportions:', prop_pred[:,0])

#score model
model_score = decon_model.score(x=X,y=Y, cell_types=cell_types)
print('Score:', model_score)