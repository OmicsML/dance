python examples/tuning/cta_svm/main.py --config_dir=config_yamls/subset_0_  --tune_mode=params --count=10 > temp_data/8.log 2>&1 &
python examples/tuning/cta_svm/main.py --config_dir=config_yamls/subset_1_  --tune_mode=params --count=10 > temp_data/9.log 2>&1 &
python examples/tuning/cta_svm/main.py --config_dir=config_yamls/subset_2_  --tune_mode=params --count=10 > temp_data/10.log 2>&1 &
