import warnings

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from dance.transforms.preprocess import findClassyGenes, ptGetTop, query_transform


class SingleCellNet():
    """Build the SingleCellNet model.

    Parameters
    ----------
    cgenesA: np.array
        All classification genes
    xpairs: np.array
        All top gene pairs
    tspRF: RandomForestClassifier
        Initialized Random Forest Classifier

    """

    def __init__(self, cgenesA=None, xpairs=None, tspRF=None):
        self.cgenesA = cgenesA
        self.xpairs = xpairs
        self.tspRF = tspRF

    def randomize(self, expDat, num=50):
        """Randomization sampling.

        Parameters
        ----------
        expDat:
            Data to be shuffled
        num: int optional
            number of samples selected

        Returns
        ----------
        output: pd.DataFrame
            A DataFrame with "num" random selected of rows

        """
        temp = expDat.to_numpy()
        temp = np.array([np.random.choice(x, len(x), replace=False) for x in temp])
        temp = temp.T
        temp = np.array([np.random.choice(x, len(x), replace=False) for x in temp]).T
        return pd.DataFrame(data=temp, columns=expDat.columns).iloc[0:num, :]

    def sc_trans_rnaseq(self, aDat, total=10000):
        """normalization to aDat.

        Parameters
        ----------
        expDat: Union[AnnData, ndarray, spmatrix]
            Data to be shuffled
        total: float optional
            If None, after normalization, each cell has a total count equal to the median of the counts_per_cell before
            normalization.

        Return
        ----------
        aDat: AnnData
            transformer result

        """
        sc.pp.normalize_per_cell(aDat, counts_per_cell_after=total)
        sc.pp.log1p(aDat)
        sc.pp.scale(aDat, max_value=10)
        return aDat

    def sc_makeClassifier(self, expTrain, genes, groups, nRand=70, ntrees=2000, stratify=False):
        """Build the random forest classifier.

        Parameters
        ----------
        expTrain: pd.DataFrame
            input dataset
        genes: np.array
            genes selected to fit the random forest model
        groups: str
            cell type labels with respect to the input
        nRand: int optional
            the number of samples selected
        ntrees: int optional
            Number of decision trees in random forest
        stratify: bool optional
            whether we select balanced class weight in the random forest model

        Returns
        ----------
        output:
            A random forest classifier

        """

        randDat = self.randomize(expTrain, num=nRand)
        expT = pd.concat([expTrain, randDat])
        allgenes = expT.columns.values
        ggenes = np.intersect1d(np.unique(genes), allgenes)
        if not stratify:
            clf = RandomForestClassifier(n_estimators=ntrees, random_state=100)
        else:
            clf = RandomForestClassifier(n_estimators=ntrees, class_weight="balanced", random_state=100)
        ggroups = np.append(np.array(groups), np.repeat("rand", nRand)).flatten()
        clf.fit(expT.loc[:, ggenes].to_numpy(), ggroups)
        return clf

    def fit(self, aTrain, dLevel, nTopGenes=100, nTopGenePairs=100, nRand=100, nTrees=1000, stratify=False,
            counts_per_cell_after=1e4, scaleMax=10, limitToHVG=False):
        """Train the SingleCellNet model(random forest)

        Parameters
        ----------
        aTrain: AnnData object
            Training set in AnnData object version
        dLevel: str
            The column name of the labels
        nTopGenes: int optional
            Top n genes selected during training
        nTopGenePairs: int optional
            Top n genes pairs selected during training
        nRand: int optional
            Samples for each training batch
        nTrees: int optional
            Number of decision trees
        stratify: bool optional
            whether we select balanced class weight in the random forest model
        counts_per_cell_after: float optional
            If None, after normalization, each cell has a total count equal to the median of the counts_per_cell before
            normalization.
        scaleMax: int optional
            Maximum scaling for data normalization
        limitToHVG: bool optional
            whether we limit the model on highly variable genes

        Returns
        ----------
        There will be no output, but the parameters in the model will be updated

        """
        warnings.filterwarnings('ignore')

        expRaw = pd.DataFrame(data=aTrain.X, index=aTrain.obs.index.values, columns=aTrain.var.index.values)
        expRaw = expRaw.loc[aTrain.obs_names]

        adNorm = aTrain.copy()
        sc.pp.normalize_per_cell(adNorm, counts_per_cell_after=counts_per_cell_after)
        sc.pp.log1p(adNorm)

        print("HVG")
        if limitToHVG:
            sc.pp.highly_variable_genes(adNorm, min_mean=0.0125, max_mean=4, min_disp=0.5)
            adNorm = adNorm[:, adNorm.var.highly_variable]

        sc.pp.scale(adNorm, max_value=scaleMax)
        expTnorm = pd.DataFrame(data=adNorm.X, index=adNorm.obs.index.values, columns=adNorm.var.index.values)
        expTnorm = expTnorm.loc[aTrain.obs_names]

        print("Matrix normalized")
        cgenesA, grps, cgenes_list = findClassyGenes(expTnorm, aTrain.obsm[dLevel], topX=nTopGenes)
        print("There are ", len(cgenesA), " classification genes\n")
        xpairs = ptGetTop(expTnorm.loc[:, cgenesA], grps, cgenes_list, topX=nTopGenePairs, sliceSize=5000)

        print("There are", len(xpairs), "top gene pairs\n")
        pdTrain = query_transform(expRaw.loc[:, cgenesA], xpairs)
        print("Finished pair transforming the data\n")
        tspRF = self.sc_makeClassifier(pdTrain.loc[:, xpairs], genes=xpairs, groups=grps, nRand=nRand, ntrees=nTrees,
                                       stratify=stratify)

        self.cgenesA = cgenesA
        self.xpairs = xpairs
        self.tspRF = tspRF

    def predict_proba(self, adata):
        """Do prediction with singlecellnet model.

        Parameters
        ----------
        adata: Union[AnnData, ndarray, spmatrix]
            test data

        Returns
        ----------
        output:
            a certain adata class from scanpy with an extra column names SCN_class as prediction

        """
        cgenes = self.cgenesA
        xpairs = self.xpairs
        rf_tsp = self.tspRF
        pred_prob = self.scn_predict(cgenes, xpairs, rf_tsp, adata, nrand=0).values
        return pred_prob

    def predict(self, adata):
        pred_prob = self.predict_proba(adata)
        pred = pred_prob.argmax(1)
        return pred

    def score(self, pred, true):
        """Compute model performance on test datasets based on accuracy.

        Parameters
        ----------
        pred
            Predicted cell-labels as a 1-d numpy array.
        true
            True cell-labels (could contain multiple cell-type per cell).

        Returns
        -------
        float
            Accuracy score of the prediction

        """
        if true.max() == 1:
            num_samples, num_classes = true.shape
            mask = pred < num_classes  # last column for unsure cells
            return true[mask, pred[mask]].sum() / num_samples
        else:
            return accuracy_score(pred, true)

    def scn_predict(self, cgenes, xpairs, rf_tsp, aDat, nrand=2):
        """Prediction with random forest.

        Parameters
        ----------
        cgenes: np.array
            Classification genes in the model
        xpairs: np.array
            Top gene pairs in the model
        rf_tsp: RandomForestClassifier
            Initialized Random Forest Classifier
        aDat: AnnData
            input feature
        nrand: int optional
            batch size during training

        Return
        ----------
        classRes_val: pd.DataFrame
            prediction result

        """
        if isinstance(aDat.X, np.ndarray):
            # in the case of aDat.X is a numpy array
            aDat.X = ad._core.views.ArrayView(aDat.X)

        expDat = pd.DataFrame(data=aDat.X.toarray(), index=aDat.obs.index.values, columns=aDat.var.index.values)
        expValTrans = query_transform(expDat.reindex(labels=cgenes, axis='columns', fill_value=0), xpairs)
        classRes_val = self.rf_classPredict(rf_tsp, expValTrans, numRand=nrand)
        return classRes_val

    def rf_classPredict(self, rfObj, expQuery, numRand=50):
        """Prediction with random forest.

        Parameters
        ----------
        rfObj:
            result of running sc_makeClassifier
        expQuery: np.array
            input features
        numRand: int optional
            batch size during training

        Return
        ----------
        xpreds: pd.DataFrame
            prediction result

        """
        if numRand > 0:
            randDat = self.randomize(expQuery, num=numRand)
            expQuery = pd.concat([expQuery, randDat])
        xpreds = pd.DataFrame(rfObj.predict_proba(expQuery), columns=rfObj.classes_, index=expQuery.index)
        return xpreds
