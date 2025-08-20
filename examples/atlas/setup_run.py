import argparse
import os
import shutil
import sys

import pandas as pd
import yaml

from dance.settings import DANCEDIR
from dance.utils import logger


def load_commands(config_path):
    """Load YAML configuration file containing command templates for different
    algorithms."""
    with open(config_path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_run_configs(run_config_path):
    """Load CSV file containing run configurations for different experiments."""
    return pd.read_csv(run_config_path)


def main():
    parser = argparse.ArgumentParser(description='Setup run parameters')
    parser.add_argument('--config', type=str, default="config/run_config.csv", help='Run configuration CSV file')

    args = parser.parse_args()

    # Load configuration files
    run_configs_df = load_run_configs(args.config)
    commands_config = load_commands("config/commands.yaml")

    # Process each run configuration
    for _, run in run_configs_df.iterrows():
        # Extract parameters for current run
        algorithm_name = run['algorithm_name']
        dataset_id = run['dataset_id']
        species = run['species']
        tissue = run['tissue']
        filetype = run['filetype']
        count = run['count']
        device = run['device']

        # Setup directory structure for the algorithm configuration
        template_path = os.path.join("config/atlas_template_yamls",
                                     f"{algorithm_name}/pipeline_params_tuning_config.yaml")
        config_dir = f"{DANCEDIR}/examples/tuning/{algorithm_name}/{dataset_id}"

        # Create configuration directory if it doesn't exist
        try:
            os.makedirs(config_dir, exist_ok=False)
        except FileExistsError:
            logger.warning(f"Error: Directory {config_dir} already exists. Please remove it before running again.")
            continue

        config_filename = f"pipeline_params_tuning_config.yaml"
        config_path = os.path.join(config_dir, config_filename)

        # Copy configuration file
        shutil.copy(template_path, config_path)
        print(f"Template copied to {config_path}")

        # Validate algorithm exists in commands configuration
        if algorithm_name not in commands_config.get("algorithms", {}):
            print(f"Error: Command not found for algorithm '{algorithm_name}'. Please check commands.yaml file.")
            continue

        # Format command template with run parameters
        command_template = commands_config["algorithms"][algorithm_name]["command"]
        run_command = command_template.format(dataset_id=dataset_id, species=species, tissue=tissue, filetype=filetype,
                                              count=count, device=device)

        # Append generated command to run script
        run_sh_path = f"{DANCEDIR}/examples/tuning/{algorithm_name}/run.sh"
        with open(run_sh_path, "a", encoding='utf-8') as run_script:
            run_script.write(f"{run_command}\n")

        print(f"Run command appended to {run_sh_path}: {run_command}")

    print("All run configurations have been processed.")


if __name__ == "__main__":
    main()
