Step Functions Index
====================

This page lists the preprocessing functions included in each step.

filter.gene
-----------

- :class:`dance.transforms.FilterGenesPercentile`
- :class:`dance.transforms.FilterGenesScanpyOrder`
- :class:`dance.transforms.FilterGenesPlaceHolder`

filter.cell
-----------

- :class:`dance.transforms.FilterCellsScanpyOrder`
- :class:`dance.transforms.FilterCellsPlaceHolder`
- :class:`dance.transforms.FilterCellsCommonMod`

normalize
---------

- :class:`dance.transforms.ColumnSumNormalize`
- :class:`dance.transforms.ScTransform`
- :class:`dance.transforms.Log1P`
- :class:`dance.transforms.NormalizeTotal`
- :class:`dance.transforms.NormalizeTotalLog1P`
- :class:`dance.transforms.tfidfTransform`
- :class:`dance.transforms.NormalizePlaceHolder`

filter.gene(highly_variable)
-----------

- :class:`dance.transforms.FilterGenesTopK`
- :class:`dance.transforms.FilterGenesRegression`
- :class:`dance.transforms.FilterGenesMatch`
- :class:`dance.transforms.HighlyVariableGenesRawCount`
- :class:`dance.transforms.HighlyVariableGenesLogarithmizedByTopGenes`
- :class:`dance.transforms.HighlyVariableGenesLogarithmizedByMeanAndDisp`
- :class:`dance.transforms.FilterGenesNumberPlaceHolder`

feature.cell
------------

- :class:`dance.transforms.CellPCA`
- :class:`dance.transforms.CellSVD`
- :class:`dance.transforms.CellSparsePCA`
- :class:`dance.transforms.WeightedFeaturePCA`
- :class:`dance.transforms.WeightedFeatureSVD`
- :class:`dance.transforms.GaussRandProjFeature`
- :class:`dance.transforms.FeatureCellPlaceHolder`