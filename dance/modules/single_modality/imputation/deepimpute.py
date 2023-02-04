"""Implementation of DeepImpute.

Extended from https://github.com/lanagarmire/DeepImpute

Reference
----------
Arisdakessian, Cédric, et al. "DeepImpute: an accurate, fast, and scalable deep neural network method to impute
single-cell RNA-seq data." Genome biology 20.1 (2019): 1-14.

"""

import tempfile
from math import floor
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import adjusted_rand_score
from torch.utils.data import DataLoader, Dataset


class NeuralNetworkModel(nn.Module):
    """ model class.
        Parameters
        ----------
        None
        Returns
        -------
        None
    """

    def __init__(self, inputdim, sub_outputdim, hidden_dim=None, dropout=0.2):
        super().__init__()
        if (hidden_dim is None):
            hidden_dim = floor(sub_outputdim / 2)
        print("hidden_dim: %f" % (hidden_dim))
        self.layer1 = nn.Linear(inputdim, hidden_dim)
        self.layer2 = nn.Dropout(p=dropout)
        self.layer3 = nn.Linear(hidden_dim, sub_outputdim)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        x = F.softplus(self.layer3(x))
        return (x)


class DeepImpute():
    """DeepImpute class.
    Parameters
    ----------
    dl_params :
        parameters including input information
    learning_rate : float optional
        learning rate
    batch_size : int optional
        batch size
    max_epochs : int optional
        maximum epochs

    patience : int optional
        number of epochs before stopping once loss stops to improve
    gpu : int optional
        option to use gpu
    loss : string optional
        loss function
    output_prefix : string optinal
        directory to save outputs
    sub_outputdim : int optional
        output dimensions in each subnetwork
    hidden_dim : int optional
        dimension of the dense layer in each subnetwork

    verbose: int optional
        verbose option

    seed: int optional
        random seed
    architecture: optional
        network architecture

    imputed_only: boolean optional
        whether to return imputed genes only

    policy: string optional
        imputation policy
    """

    def __init__(self, dl_params, output_prefix=tempfile.mkdtemp(), architecture=None):
        self.architecture = architecture
        self.outputdir = output_prefix
        self.dl_params = dl_params

    def loadDefaultArchitecture(self):
        """load default model architecture.
        Parameters
        ----------
        None
        Returns
        -------
        None
        """
        
        if (self.hidden_dim is None):
            hidden_dim = self.sub_outputdim // 2
        else:
            hidden_dim = self.hidden_dim
        self.NN_parameters['architecture'] = [
            {
                "type": "dense",
                "neurons": hidden_dim,
                "activation": "relu"
            },
            {
                "type": "dropout",
                "rate": 0.2
            },
        ]

    def wMSE(self, y_true, y_pred, binary=False):
        """weighted MSE.
        Parameters
        ----------
        y_true: array
            true expression
        Y_train: array
            predicted expression
        binary: boolean optional
            whether to use binary weights
        Returns
        -------
        val: float
            weighted MSE
        """

        if binary:
            tmp = y_true > 0
            weights = tmp.type(torch.FloatTensor)
        else:
            weights = y_true
        val = torch.mean(weights * torch.square(y_true - y_pred))
        return val

    def build(self, inputdims):
        """build model.
        Parameters
        ----------
        inputdims: int
            number of neurons as input in the first layer
        Returns
        -------
        models : array
            array of subnetworks
        """

        self.NN_parameters = { 
            "learning_rate": self.dl_params.lr,
            "batch_size": self.dl_params.batch_size,
	    "loss": self.dl_params.loss,
	    "architecture": self.architecture,
	    "max_epochs": self.dl_params.n_epochs,
            "patience": self.dl_params.patience
	}
        self.sub_outputdim = self.dl_params.sub_outputdim
        self.hidden_dim = self.dl_params.hidden_dim

        if self.NN_parameters['architecture'] is None:
            self.loadDefaultArchitecture()

        print(self.NN_parameters['architecture'])

        models = []
        for dim in inputdims:
            models.append(
                NeuralNetworkModel(dim, self.sub_outputdim, hidden_dim=self.hidden_dim,
                                   dropout=self.NN_parameters['architecture'][1]['rate']))

        return models

    def fit(self, X_train=None, Y_train=None, X_test=None, Y_test=None, inputgenes=None):
        """Train model.
        Parameters
        ----------
        X_train: optional
            Training data including input genes
        Y_train: optional
            Training data including target genes to be inputed
        X_test:  optional
            Validation data including input predictor genes
        Y_test:  optional
            Validation data including target genes to be inputed
        inputgenes: array optional
            input genes as predictors for target genes
        Returns
        -------
        None
        """
        for attr in ['X_train', 'Y_train', 'X_test', 'Y_test', 'inputgenes', 'batch_size', 'gpu']:
            setattr(self, attr, getattr(self.dl_params, attr))
        self.device = torch.device('cuda' if self.gpu != -1 and torch.cuda.is_available() else 'cpu')

        if (X_train is None):
            X_train = self.X_train
        if (Y_train is None):
            Y_train = self.Y_train
        if (X_test is None):
            X_test = self.X_test
        if (Y_test is None):
            Y_test = self.Y_test
        if (inputgenes is None):
            inputgenes = self.inputgenes

        device = self.device
        print("Building network")
        self.models = self.build([len(genes) for genes in inputgenes])

        data = [torch.cat((torch.Tensor(X_train[i]), torch.Tensor(Y_train[i])), dim=1) for i in range(len(X_train))]
        train_loaders = [DataLoader(data[i], batch_size=self.batch_size, shuffle=True) for i in range(len(data))]
        for i, model in enumerate(self.models):
            optimizer = optim.Adam(model.parameters(), lr=self.NN_parameters['learning_rate'])
            val_losses = []
            for epoch in range(self.NN_parameters['max_epochs']):
                epoch_loss = 0
                train_loader = train_loaders[i]
                model.train()
                for batch_idx, data in enumerate(train_loader):
                    X_batch = data[:, :(data.shape[1] - self.sub_outputdim)].to(device)
                    Y_batch = data[:, (data.shape[1] - self.sub_outputdim):].to(device)
                    model.to(device)
                    y_pred = model.forward(X_batch)
                    loss = self.wMSE(Y_batch, y_pred)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                average_epoch_loss = epoch_loss / len(train_loader)
                X_test_i = torch.Tensor(X_test[i])
                Y_test_i = torch.Tensor(Y_test[i])
                val_loss, _ = self.eval(X_test_i, Y_test_i, model)
                val_losses.append(val_loss.item())
                min_val = min(val_losses)
                patience = self.NN_parameters['patience']
                if val_loss == min_val:
                    #self.save_model(model, optimizer, i)
                    self.models[i] = model
                    print('Saving model %s.' % i)
                print("Average epoch loss: {:f}, epoch eval loss: {:f}.".format(average_epoch_loss, val_loss))
                # if epoch >= patience and min(val_losses[-patience:]) > min_val:
                #     print("Early stopping on epoch %s." % epoch)
                #     model = self.load_model(model, i)
                #     break
            #self.models[i] = model

    def save_model(self, model, optimizer, i):
        """save model.
        Parameters
        ----------
        model:
            model to be saved

        optimizer:
            optimizer

        i: int
            index of the subnetwork to be loaded
        Returns
        -------
        None
        """
        self.prj_path = Path(
            __file__).parent.resolve().parent.resolve().parent.resolve().parent.resolve().parent.resolve()
        self.save_path = self.prj_path / 'example' / 'single_modality' / 'imputation' / 'pretrained' / self.dl_params.train_dataset / 'models'
        if not self.save_path.exists():
            self.save_path.mkdir(parents=True)

        model_string = 'model_' + str(i)
        opt_string = 'optimizer_' + str(i)
        state = {model_string: model.state_dict(), opt_string: optimizer.state_dict()}
        torch.save(state, self.save_path / f"{self.dl_params.train_dataset}_{i}.pt")

    def load_model(self, model, i):
        """load model.
        Parameters
        ----------
        model:
            model to be loaded

        i: int
            index of the subnetwork to be loaded
        Returns
        -------
        model :
            loaded model
        """

        model_path = self.save_path / f"{self.dl_params.train_dataset}_{i}.pt"
        state = torch.load(model_path, map_location=self.device)
        model_string = 'model_' + str(i)
        model.load_state_dict(state[model_string])
        return model

    def predict(self, targetgenes=None):
        """Get predictions from the trained model.
        Parameters
        ----------
        targetgenes: array optional
            genes to be imputed
        Returns
        -------
        imputed : DataFrame
            imputed gene expression
        """
        for attr in ['targetgenes', 'total_counts', 'policy', 'imputed_only']:
            setattr(self, attr, getattr(self.dl_params, attr))
        if (targetgenes is None):
            targetgenes = self.targetgenes

        device = self.device

        norm_data = np.log1p(self.total_counts)  # is norm_raw a saved variable?

        # select input data
        X_data = [torch.Tensor(norm_data.loc[:, inputgenes].values).to(device) for inputgenes in self.inputgenes]

        # make predictions using each subnetwork
        Y_pred_lst = []
        for i, model in enumerate(self.models):
            model.eval()
            with torch.no_grad():
                Y_pred_lst.append(model.forward(X_data[i]))
        # concatenate predicted values
        Y_pred = torch.cat(Y_pred_lst, dim=1)
        # convert to pd
        Y_pred = pd.DataFrame(Y_pred.cpu().numpy(), index=self.total_counts.index, columns=targetgenes.flatten())

        # create imputed matrix
        predicted = Y_pred.groupby(by=Y_pred.columns, axis=1).mean()
        not_predicted = norm_data.drop(targetgenes.flatten(), axis=1)
        imputed = (pd.concat([predicted, not_predicted], axis=1).loc[self.total_counts.index,
                                                                     self.total_counts.columns].values)

        # To prevent overflow
        imputed[(imputed > 2 * norm_data.values.max()) | (np.isnan(imputed))] = 0
        # Convert back to counts
        imputed = np.expm1(imputed)

        if self.policy == "restore":
            print("Filling zeros")
            mask = (self.total_counts.values > 0)
            imputed[mask] = self.total_counts.values[mask]
        elif self.policy == "max":
            print("Imputing data with 'max' policy")
            mask = (self.total_counts.values > imputed)
            imputed[mask] = self.total_counts.values[mask]

        imputed = pd.DataFrame(imputed, index=self.total_counts.index, columns=self.total_counts.columns)

        if self.imputed_only:
            return imputed.loc[:, predicted.columns]
        else:
            return imputed

    def score(self, true_expr, true_labels=None, n_neighbors=None, n_pcs=None, clu_resolution=1, targetgenes=None):
        """Evaluate the trained model.
        Parameters
        ----------
        true_expr : DataFrame
            true expression
        true_labels : array optional
            provided cell labels
        n_neighbors : int optional
            number of neighbors tp cluster imputed data
        n_pcs: int optional
            number of principal components for neighbor detection
        clu_resolution : int optional
            resolution for Leiden cluster
        targetgenes: array optional
            genes to be imputed
        Returns
        -------
        ari : float
            adjusted Rand index.
        MSE_cells :
            mean squared errors averaged across genes per cell
        MSE_genes :
            mean squared errors averaged across cellss per gene
        """
        self.true_counts = self.dl_params.true_counts

        if (targetgenes == None):  # set target genes
            targetgenes = self.targetgenes
        imputed_expr = self.predict(targetgenes=targetgenes)  # prediction

        # subset target genes only
        targetgenes = targetgenes.flatten()  # flatten target genes list
        imputed_expr = imputed_expr.loc[:, targetgenes]
        imputed_expr = imputed_expr.groupby(by=imputed_expr.columns, axis=1).mean()

        true_expr = self.true_counts
        masked_expr = self.total_counts
        mask = (masked_expr != true_expr)
        df_deepImpute = np.log1p(imputed_expr)
        gene_subset = df_deepImpute.columns
        true_expr = np.log1p(true_expr.reindex(columns=gene_subset))
        masked_expr = np.log1p(masked_expr.reindex(columns=gene_subset))
        mask = mask.reindex(columns=gene_subset)
        MSE_cells = pd.DataFrame(((df_deepImpute[mask] - true_expr[mask])**2).mean(axis=1)).dropna()
        MSE_genes = pd.DataFrame(((df_deepImpute[mask] - true_expr[mask])**2).mean(axis=0)).dropna()
        return MSE_cells, MSE_genes

    def eval(self, X_test, Y_test, model):
        """evaluate model.
        Parameters
        ----------
        X_test:
            Validation data including input predictor genes
        Y_test:
            Validation data including target genes to be inputed

        model:
            model to be evaluated
        Returns
        -------
        loss : float
            weighted MSE as loss
        y_pred:
            predicted expression
        """
        device = torch.device('cuda' if self.gpu != -1 and torch.cuda.is_available() else 'cpu')
        model.eval()
        with torch.no_grad():
            y_pred = model.forward(X_test.to(device))
            loss = self.wMSE(Y_test.to(device), y_pred)
        return loss, y_pred

