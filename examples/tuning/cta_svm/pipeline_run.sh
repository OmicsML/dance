#!/bin/bash
log_dir="temp_data"
mkdir -p "${log_dir}"
python /home/zyxing/dance/examples/tuning/cta_svm/main.py --config_dir=/home/zyxing/dance/examples/tuning/cta_svm/config_yamls/pipeline/subset_0_ --count=1 > ${log_dir}/0.log 2>&1 &
python /home/zyxing/dance/examples/tuning/cta_svm/main.py --config_dir=/home/zyxing/dance/examples/tuning/cta_svm/config_yamls/pipeline/subset_6_ --count=28 > ${log_dir}/6.log 2>&1 &
python /home/zyxing/dance/examples/tuning/cta_svm/main.py --config_dir=/home/zyxing/dance/examples/tuning/cta_svm/config_yamls/pipeline/subset_5_ --count=16 > ${log_dir}/5.log 2>&1 &
python /home/zyxing/dance/examples/tuning/cta_svm/main.py --config_dir=/home/zyxing/dance/examples/tuning/cta_svm/config_yamls/pipeline/subset_2_ --count=7 > ${log_dir}/2.log 2>&1 &
python /home/zyxing/dance/examples/tuning/cta_svm/main.py --config_dir=/home/zyxing/dance/examples/tuning/cta_svm/config_yamls/pipeline/subset_4_ --count=28 > ${log_dir}/4.log 2>&1 &
python /home/zyxing/dance/examples/tuning/cta_svm/main.py --config_dir=/home/zyxing/dance/examples/tuning/cta_svm/config_yamls/pipeline/subset_1_ --count=4 > ${log_dir}/1.log 2>&1 &
python /home/zyxing/dance/examples/tuning/cta_svm/main.py --config_dir=/home/zyxing/dance/examples/tuning/cta_svm/config_yamls/pipeline/subset_3_ --count=4 > ${log_dir}/3.log 2>&1 &
python /home/zyxing/dance/examples/tuning/cta_svm/main.py --config_dir=/home/zyxing/dance/examples/tuning/cta_svm/config_yamls/pipeline/subset_7_ --count=112 > ${log_dir}/7.log 2>&1 &
