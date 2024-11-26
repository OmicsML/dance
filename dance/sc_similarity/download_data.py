from dance.datasets.singlemodality import CellTypeAnnotationDataset


def get_anndata(tissue: str = "Blood", species: str = "human", filetype: str = "h5ad", train_dataset=[],
                test_dataset=[], valid_dataset=[], data_dir="../temp_data"):
    data = CellTypeAnnotationDataset(train_dataset=train_dataset, test_dataset=test_dataset,
                                     valid_dataset=valid_dataset, data_dir=data_dir, tissue=tissue, species=species,
                                     filetype=filetype).load_data()
    return data.data
