# python examples/tuning/cta_svm/main.py --config_dir=config_yamls/subset_0_  # don't need 4 7 4 28 16 28
python examples/tuning/cta_svm/main.py --config_dir=config_yamls/subset_1_  --count=4 > data/1.log 2>&1 &
python examples/tuning/cta_svm/main.py --config_dir=config_yamls/subset_2_  --count=7 > data/2.log 2>&1 &
python examples/tuning/cta_svm/main.py --config_dir=config_yamls/subset_3_  --count=4 > data/3.log 2>&1 &
python examples/tuning/cta_svm/main.py --config_dir=config_yamls/subset_4_  --count=28 > data/4.log 2>&1 &
python examples/tuning/cta_svm/main.py --config_dir=config_yamls/subset_5_  --count=16 > data/5.log 2>&1 &
python examples/tuning/cta_svm/main.py --config_dir=config_yamls/subset_6_  --count=28 > data/6.log 2>&1 &
python examples/tuning/cta_svm/main.py --config_dir=config_yamls/subset_7_  --count=112 > data/7.log 2>&1 &
