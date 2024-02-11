#!/bin/bash
log_dir="temp_data"
mkdir -p "${log_dir}"
python examples/tuning/cta_svm/main.py --config_dir=config_yamls/params/1st_test_acc_  --tune_mode=params --count=10 > ${log_dir}/8.log 2>&1 &
python examples/tuning/cta_svm/main.py --config_dir=config_yamls/params/2nd_test_acc_  --tune_mode=params --count=10 > ${log_dir}/9.log 2>&1 &
python examples/tuning/cta_svm/main.py --config_dir=config_yamls/params/3rd_test_acc_  --tune_mode=params --count=10 > ${log_dir}/10.log 2>&1 &
