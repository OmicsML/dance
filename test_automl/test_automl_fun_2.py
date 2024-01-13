import wandb
import yaml

ARTIFACT_FILENAME = "/home/zyxing/dance/"
ARTIFACT_NAME = "optuna-config"

PROJECT = "pytorch-cell_type_annotation_ACTINN_function_new"
ENTITY = "xzy11632"
QUEUE = "actinn"  ## Put in a Launch queue you've created and started


def train(a):
    pass


if __name__ == "__main__":
    # """create and log artifact to wandb"""
    run = wandb.init(project=PROJECT, entity=ENTITY)
    artifact = wandb.Artifact(name=ARTIFACT_NAME, type='optuna')
    artifact.add_dir(ARTIFACT_FILENAME)
    run.log_artifact(artifact)
    run.finish()
#     config = {
#     "metric": {"name": "scores", "goal": "maximize"},
#     "run_cap": 4,
#     "job": "xzy11632/pytorch-cell_type_annotation_ACTINN_function_new/job-source-pytorch-cell_type_annotation_ACTINN_function_new-test_automl_test_automl_fun_job.py:v3",#思考一下
#     "scheduler": {
#         "job": "xzy11632/pytorch-cell_type_annotation_ACTINN_function_new/job-OptunaScheduler:v4",
#         "num_workers": 2,
#         "settings": {
#             "optuna_source": f"{ENTITY}/{PROJECT}/{artifact.wait().name}",
#             "optuna_source_filename": ARTIFACT_FILENAME,
#             "resource": "local-container",  # required for scheduler jobs sourced from images

#             # optional sampler args
#             "pruner": {
#                 "type": "PercentilePruner",
#                 "args": {
#                     "percentile": 0.25,
#                     "n_startup_trials": 2,
#                     "n_min_trials": 1,  # min epochs before prune
#                 }
#             }
#         }
#     },
#     # parameters are not needed when loading a conditional config from an artifact
#     # "parameters": {
#     #     'epochs': {'values': [5, 10, 15]},
#     #     'lr': {'max': 0.1, 'min': 0.0001}
#     # }
# }
#     # write config to file
#     config_filename = "sweep-config.yaml"
#     yaml.dump(config, open(config_filename, "w"))
