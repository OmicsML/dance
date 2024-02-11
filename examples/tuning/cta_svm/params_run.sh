python examples/tuning/cta_svm/main.py --config_dir=config_yamls/1st_test_acc_  --tune_mode=params --count=10 > temp_data/8.log 2>&1 &
python examples/tuning/cta_svm/main.py --config_dir=config_yamls/2nd_test_acc_  --tune_mode=params --count=10 > temp_data/9.log 2>&1 &
python examples/tuning/cta_svm/main.py --config_dir=config_yamls/3rd_test_acc_  --tune_mode=params --count=10 > temp_data/10.log 2>&1 &
