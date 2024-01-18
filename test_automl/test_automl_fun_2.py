import yaml

import wandb

ARTIFACT_FILENAME = "/home/zyxing/dance/"
ARTIFACT_NAME = "optuna-config"

PROJECT = "pytorch-cell_type_annotation_ACTINN_function_new"
ENTITY = "xzy11632"
QUEUE = "actinn"  ## Put in a Launch queue you've created and started

if __name__ == "__main__":
    # """create and log artifact to wandb"""
    run = wandb.init(project=PROJECT, entity=ENTITY)
    artifact = wandb.Artifact(name=ARTIFACT_NAME, type='optuna')
    artifact.add_dir(ARTIFACT_FILENAME)
    run.log_artifact(artifact)
    run.finish()
