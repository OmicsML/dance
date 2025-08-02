#!/bin/bash
log_dir="temp_data"
mkdir -p "${log_dir}"
python examples/tuning/cta_svm/main.py --result_name=1st_best_test_acc.csv --config_dir=config_yamls/params/1_test_acc_  --tune_mode=params --count=10 > ${log_dir}/8.log 2>&1 &
python examples/tuning/cta_svm/main.py --result_name=2nd_best_test_acc.csv --config_dir=config_yamls/params/2_test_acc_  --tune_mode=params --count=10 > ${log_dir}/9.log 2>&1 &
python examples/tuning/cta_svm/main.py --result_name=3rd_best_test_acc.csv --config_dir=config_yamls/params/3_test_acc_  --tune_mode=params --count=10 > ${log_dir}/10.log 2>&1 &
