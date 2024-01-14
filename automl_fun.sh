cd ~/dance
python test_automl/test_automl_fun_job.py
python test_automl/test_automl_fun_2.py
cd ~
python dance/optuna_scheduler.py --entity xzy11632 --project pytorch-cell_type_annotation_ACTINN_function_new
wandb launch-sweep dance/test_automl/sweep-config.yaml -e xzy11632 -p pytorch-cell_type_annotation_ACTINN_function_new -q tutorial-run-queue
