#!/bin/bash
log_dir=temp_data
mkdir -p ${log_dir}
python examples/tuning/cta_svm/main.py --config_dir=config_yamls/pipeline/subset_0_ --count=4 > temp_data/0.log 2>&1 &
python examples/tuning/cta_svm/main.py --config_dir=config_yamls/pipeline/subset_1_ --count=7 > temp_data/1.log 2>&1 &
python examples/tuning/cta_svm/main.py --config_dir=config_yamls/pipeline/subset_2_ --count=4 > temp_data/2.log 2>&1 &
python examples/tuning/cta_svm/main.py --config_dir=config_yamls/pipeline/subset_3_ --count=28 > temp_data/3.log 2>&1 &
python examples/tuning/cta_svm/main.py --config_dir=config_yamls/pipeline/subset_4_ --count=16 > temp_data/4.log 2>&1 &
python examples/tuning/cta_svm/main.py --config_dir=config_yamls/pipeline/subset_5_ --count=28 > temp_data/5.log 2>&1 &
python examples/tuning/cta_svm/main.py --config_dir=config_yamls/pipeline/subset_6_ --count=112 > temp_data/6.log 2>&1 &
