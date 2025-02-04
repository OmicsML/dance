# How to Add New Algorithms to the Auto-Search Framework

This document explains how to integrate new algorithms into the project's automatic search framework.

## Implementation Requirements

### 1. Inherit Base Classes

New algorithms should inherit from appropriate base classes in `dance.modules.base`:

```python
from dance.modules.base import (
    BaseClusteringMethod,    # Clustering algorithms
    BaseClassificationMethod,  # Classification algorithms
    BaseRegressionMethod,    # Regression algorithms
    TorchNNPretrain         # If pretraining is needed
)

class YourMethod(BaseClusteringMethod):
    """Your method description."""
    pass
```

### 2. Implement Required Interfaces

All algorithms must implement these core interfaces:

```python
def preprocessing_pipeline(**kwargs) -> BaseTransform:
    """Define data preprocessing pipeline"""
    ...

def fit(self, x, y=None, **kwargs):
    """Train the model

    Parameters
    ----------
    x : array-like
        Input features
    y : array-like, optional
        Labels (required for supervised learning)
    """
    ...

def predict(self, x):
    """Predict results

    Parameters
    ----------
    x : array-like
        Input features

    Returns
    -------
    array-like
        Prediction results
    """
    ...
```

The base class provides a default implementation of the `score()` method, which:

1. Calls `predict()` to get prediction results
1. Calculates scores using predefined evaluation metrics
   - Clustering algorithms: ARI (Adjusted Rand Index)
   - Classification algorithms: Accuracy
   - Regression algorithms: MSE (Mean Squared Error)

## Directory Structure

New algorithms should follow this directory structure:

```plaintext
    examples/tuning/
    └── [task_name]_[algorithm_name]/
        ├── main.py # Main execution file
        └── [dataset_name]/ # Dataset related configurations
            ├── pipeline_params_tuning_config.yaml
            └── config_yamls
                ├── 0_test_acc_params_tuning_config.yaml
                ├── 1_test_acc_params_tuning_config.yaml
                └── 2_test_acc_params_tuning_config.yaml
```

## Integration Steps

### 1. Create Algorithm Directory

Create a new algorithm directory under `examples/tuning/`, named as `[task_name]_[algorithm_name]`.

### 2. Implement Main Execution File

Create `main.py` in the algorithm directory with these key components:

1. **Parameter Configuration**

```python
parser = argparse.ArgumentParser()
# Add necessary parameters
parser.add_argument("--data_dir", default="../temp_data"，help="Directory path containing the input data files")
parser.add_argument("--dataset", type=str, choices=[...]，help="Dataset name")
parser.add_argument("--tune_mode", default="pipeline_params",
                   choices=["pipeline", "params", "pipeline_params"]， help="Tuning mode: 'pipeline' for pipeline tuning only, 'params' for parameter tuning only, 'pipeline_params' for both")
parser.add_argument("--sweep_id", type=str, default=None，help="Existing sweep ID to resume. If None, creates a new sweep")
parser.add_argument("--count", type=int, default=2，help="Number of times to run the sweep agent")
parser.add_argument(
        "--summary_file_path",
        default="results/pipeline/best_test_acc.csv",
        type=str,
        help="Path to save the summary results file"
    )
parser.add_argument(
        "--root_path",
        default=str(Path(__file__).resolve().parent),
        type=str,
        help="Root directory path for saving results and configuration files"
    )
# ... other model-specific parameters ...
args = parser.parse_args()
```

2. **Evaluation Function Definition**

```python
def evaluate_pipeline(tune_mode, pipeline_planer):
    # Initialize wandb
    wandb.init(settings=wandb.Settings(start_method='thread'))

    # Load data according to the task
    dataloader = TaskDataset(args.data_dir, args.dataset)
    data = dataloader.load_data(cache=args.cache)

    # Apply preprocessing pipeline
    kwargs = {tune_mode: dict(wandb.config)}
    preprocessing_pipeline = pipeline_planer.generate(**kwargs)
    preprocessing_pipeline(data)

    # Get processed data
    inputs, y = data.get_data(return_type="default")

    # Initialize and train model
    model = YourModel(model_params)
    model.fit(inputs, y)

    # Evaluate and log results
    score = model.score(None, y)
    wandb.log({"acc": score})
    wandb.finish()
```

3. **Main Program Flow**

```python
if __name__ == "__main__":
    # Initialize pipeline planer
    pipeline_planer = PipelinePlaner.from_config_file(
        f"{file_root_path}/{args.tune_mode}_tuning_config.yaml")

    # Run hyperparameter search
    entity, project, sweep_id = pipeline_planer.wandb_sweep_agent(
        evaluate_pipeline,
        sweep_id=args.sweep_id,
        count=args.count
    )

    # Save results
    save_summary_data(
        entity, project, sweep_id,
        summary_file_path=args.summary_file_path,
        root_path=file_root_path
    )
    if args.tune_mode == "pipeline" or args.tune_mode == "pipeline_params":
      #generate step3_default_params.yaml
        get_step3_yaml(result_load_path=f"{args.summary_file_path}", step2_pipeline_planer=pipeline_planer,
                       conf_load_path=f"{Path(args.root_path).resolve().parent}/step3_default_params.yaml",
                       root_path=file_root_path, required_funs=["SaveRaw", "UpdateRaw", "NeighborGraph", "SetConfig"],
                       required_indexes=[2, 5, sys.maxsize - 1, sys.maxsize], metric="acc")
        if args.tune_mode == "pipeline_params":
            #run step3
            run_step3(file_root_path, evaluate_pipeline, tune_mode="params", step2_pipeline_planer=pipeline_planer)
```

### 3. Configuration File Setup

Create corresponding configuration files in the dataset directory to guide the hyperparameter search process.

#### Configuration File Types

- `pipeline_params_tuning_config.yaml`: Main configuration file for joint search
- `config_yamls/*.yaml`: Parameter search configuration files automatically generated by the system

#### Search Modes Explanation

The system supports three search modes (specified by `tune_mode`):

1. **pipeline mode**

   - Only searches for optimal preprocessing pipeline combinations
   - Uses `pipeline_tuning_config.yaml`

1. **params mode**

   - Only searches for optimal model parameter combinations
   - Uses `params_tuning_config.yaml`

1. **pipeline_params mode**

   - Performs two-stage joint search
   - First stage: searches for optimal preprocessing pipeline
   - Second stage: searches for optimal model parameters based on the best pipeline
   - System automatically generates parameter search config files (e.g., `config_yamls/0_test_acc_params_tuning_config.yaml`)

Configuration file example:

```yaml
# pipeline_params_tuning_config.yaml
type: preprocessor
tune_mode: pipeline_params
pipeline_tuning_top_k: 3 #topk for pipeline tuning to use parameter tuning
parameter_tuning_freq_n: 20 #frequency for parameter tuning
pipeline:
  - type: filter.gene
    include:
      - FilterGenesPercentile
      - FilterGenesScanpyOrder
      - FilterGenesPlaceHolder
    default_params:
      FilterGenesScanpyOrder:
        order: ["min_counts", "min_cells", "max_counts", "max_cells"]
        min_counts: 0.01
        max_counts: 0.99
        min_cells: 0.01
        max_cells: 0.99
  - type: feature.cell
    include:
      - WeightedFeaturePCA
      - WeightedFeatureSVD
      - CellPCA
      - CellSVD
      - GaussRandProjFeature  # Registered custom preprocessing func
      - FeatureCellPlaceHolder
    params:
      out: feature.cell
      log_level: INFO
    default_params:
      WeightedFeaturePCA:
        split_name: train
      WeightedFeatureSVD:
        split_name: train
  - type: misc
    target: SetConfig
    params:
      config_dict:
        feature_channel: feature.cell
        label_channel: cell_type
wandb:
  entity: xxxxx
  project: xxxxx
  method: grid #try grid to provide a comprehensive search
  metric:
    name: acc  # val/acc
    goal: maximize
```

### 4. Run Tests

After integration, test using these commands:

```bash
    # Search preprocessing pipeline only
    python main.py --tune_mode pipeline
    # Search model parameters only
    python main.py --tune_mode params
    # Joint search
    python main.py --tune_mode pipeline_params
```

## Notes

1. Ensure the model implements `fit()` and `score()` interfaces
1. wandb configuration should correspond to model parameters
1. Recommend testing on small datasets first

## Examples

Refer to `examples/tuning/cluster_graphsc` and `examples/tuning/cta_celltypist` implementations.
