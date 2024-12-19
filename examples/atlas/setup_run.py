import argparse
import os
import shutil
import sys

import yaml

from dance.settings import DANCEDIR


def load_commands(config_path):
    with open(config_path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_run_configs(run_config_path):
    with open(run_config_path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Setup run parameters')
    parser.add_argument('--config', type=str, default="config/run_config.yaml", help='Run configuration YAML file')

    args = parser.parse_args()

    run_configs = load_run_configs(args.config)

    commands_config = load_commands("commands.yaml")

    for run in run_configs.get("runs", []):
        algorithm_name = run.get('algorithm_name')
        dataset_id = run.get('dataset_id')
        species = run.get('species')
        tissue = run.get('tissue')
        filetype = run.get('filetype')
        count = run.get('count')
        device = run.get('device')

        # Define paths
        template_path = os.path.join("config/atlas_template_yamls",
                                     f"{algorithm_name}/pipeline_params_tuning_config.yaml")
        config_dir = f"{DANCEDIR}/examples/tuning/{algorithm_name}/{dataset_id}"
        os.makedirs(config_dir, exist_ok=True)
        config_filename = f"pipeline_params_tuning_config.yaml"
        config_path = os.path.join(config_dir, config_filename)

        # Copy configuration file
        shutil.copy(template_path, config_path)
        print(f"Template copied to {config_path}")

        if algorithm_name not in commands_config.get("algorithms", {}):
            print(f"Error: Command not found for algorithm '{algorithm_name}'. Please check commands.yaml file.")
            continue

        command_template = commands_config["algorithms"][algorithm_name]["command"]
        run_command = command_template.format(dataset_id=dataset_id, species=species, tissue=tissue, filetype=filetype,
                                              count=count, device=device)

        # Append the run command to run.sh
        run_sh_path = f"{DANCEDIR}/examples/tuning/{algorithm_name}/run.sh"
        with open(run_sh_path, "a", encoding='utf-8') as run_script:
            run_script.write(f"{run_command}\n")

        print(f"Run command appended to {run_sh_path}: {run_command}")

    print("All run configurations have been processed.")


if __name__ == "__main__":
    main()
