


python actinn.py --species human --tissue Immune --train_dataset 11407 1519 636 713 9054 9258 --test_dataset 1925 205 3323 6509 7572  --device cuda:2 --num_runs 5 > 11407-1519-636-713-9054-9258_1925-205-3323-6509-7572-actinn-out.log 2>&1 &
python celltypist.py --species human --tissue Immune --train_dataset 11407 1519 636 713 9054 9258 --test_dataset 1925 205 3323 6509 7572 --num_runs 5 > 11407-1519-636-713-9054-9258_1925-205-3323-6509-7572-celltypist-out.log 2>&1 &
python scdeepsort.py --species human --tissue Immune --train_dataset 11407 1519 636 713 9054 9258 --test_dataset 1925 205 3323 6509 7572  --device cuda:5 --num_runs 5 > 11407-1519-636-713-9054-9258_1925-205-3323-6509-7572-scdeepsort-out.log 2>&1 &
python singlecellnet.py --species human --tissue Immune --train_dataset 11407 1519 636 713 9054 9258 --test_dataset 1925 205 3323 6509 7572 --num_runs 5 > 11407-1519-636-713-9054-9258_1925-205-3323-6509-7572-singlecellnet-out.log 2>&1 &
python svm.py --species human --tissue Immune --train_dataset 11407 1519 636 713 9054 9258 --test_dataset 1925 205 3323 6509 7572  --gpu 6 --num_runs 5 > 11407-1519-636-713-9054-9258_1925-205-3323-6509-7572-svm-out.log 2>&1 &


python actinn.py --species human --tissue Spleen --train_dataset 3043 3777 4029 4115 4362 4657 --test_dataset 1729 2125 2184 2724 2743 --device cuda:1 --num_runs 5 > 3043-3777-4029-4115-4362-4657_1729-2125-2184-2724-2743-actinn-out.log 2>&1 &
python celltypist.py --species human --tissue Spleen --train_dataset 3043 3777 4029 4115 4362 4657 --test_dataset 1729 2125 2184 2724 2743 --num_runs 5 > 3043-3777-4029-4115-4362-4657_1729-2125-2184-2724-2743-celltypist-out.log 2>&1 &
python scdeepsort.py --species human --tissue Spleen --train_dataset 3043 3777 4029 4115 4362 4657 --test_dataset 1729 2125 2184 2724 2743 --device cuda:4 --num_runs 5 > 3043-3777-4029-4115-4362-4657_1729-2125-2184-2724-2743-scdeepsort-out.log 2>&1 &
python singlecellnet.py --species human --tissue Spleen --train_dataset 3043 3777 4029 4115 4362 4657 --test_dataset 1729 2125 2184 2724 2743 --num_runs 5 > 3043-3777-4029-4115-4362-4657_1729-2125-2184-2724-2743-singlecellnet-out.log 2>&1 &
python svm.py --species human --tissue Spleen --train_dataset 3043 3777 4029 4115 4362 4657 --test_dataset 1729 2125 2184 2724 2743  --gpu 6 --num_runs 5 > 3043-3777-4029-4115-4362-4657_1729-2125-2184-2724-2743-svm-out.log 2>&1 &

python actinn.py --species human --tissue CD8 --train_dataset 1027 1357 1641 517 706 777 850 972 --test_dataset 245 332 377 398 405 455 470 492 --device cuda:2 --num_runs 5 > 1027-1357-1641-517-706-777-850-972_245-332-377-398-405-455-470-492-actinn-out.log 2>&1 &
python celltypist.py --species human --tissue CD8 --train_dataset 1027 1357 1641 517 706 777 850 972 --test_dataset 245 332 377 398 405 455 470 492 --num_runs 5 > 1027-1357-1641-517-706-777-850-972_245-332-377-398-405-455-470-492-celltypist-out.log 2>&1 &
python scdeepsort.py --species human --tissue CD8 --train_dataset 1027 1357 1641 517 706 777 850 972 --test_dataset 245 332 377 398 405 455 470 492 --device cuda:3 --num_runs 5 > 1027-1357-1641-517-706-777-850-972_245-332-377-398-405-455-470-492-scdeepsort-out.log 2>&1 &
python singlecellnet.py --species human --tissue CD8 --train_dataset 1027 1357 1641 517 706 777 850 972 --test_dataset 245 332 377 398 405 455 470 492 --num_runs 5 > 1027-1357-1641-517-706-777-850-972_245-332-377-398-405-455-470-492-singlecellnet-out.log 2>&1 &
python svm.py --species human --tissue CD8 --train_dataset 1027 1357 1641 517 706 777 850 972 --test_dataset 245 332 377 398 405 455 470 492  --gpu 0 --num_runs 5 > 1027-1357-1641-517-706-777-850-972_245-332-377-398-405-455-470-492-svm-out.log 2>&1 &


python actinn.py --species human --tissue Brain --train_dataset 328 --test_dataset 138 --device cuda:1 --num_runs 5 > 328-138-actinn-out.log 2>&1 &
python celltypist.py --species human --tissue Brain --train_dataset 328 --test_dataset 138 --num_runs 5 > 328-138-celltypist-out.log 2>&1 &
python scdeepsort.py --species human --tissue Brain --train_dataset 328 --test_dataset 138 --device cuda:2 --num_runs 5 --dense_dim 300 > 328-138-scdeepsort-out.log 2>&1 &
python singlecellnet.py --species human --tissue Brain --train_dataset 328 --test_dataset 138 --num_runs 5 > 328-138-singlecellnet-out.log 2>&1 &
python svm.py --species human --tissue Brain --train_dataset 328 --test_dataset 138  --gpu 0 --num_runs 5 --dense_dim 300 > 328-138-svm-out.log 2>&1 &

python actinn.py --species human --tissue CD4 --train_dataset 1013 1247 598 732 767 768 770 784 845 864 --test_dataset 315 340 376 381 390 404 437 490 551 559 --device cuda:1 --num_runs 5 > 1013-1247-598-732-767-768-770-784-845-864_315-340-376-381-390-404-437-490-551-559-actinn-out.log 2>&1 &
python celltypist.py --species human --tissue CD4 --train_dataset 1013 1247 598 732 767 768 770 784 845 864 --test_dataset 315 340 376 381 390 404 437 490 551 559 --num_runs 5 > 1013-1247-598-732-767-768-770-784-845-864_315-340-376-381-390-404-437-490-551-559-celltypist-out.log 2>&1 &
python scdeepsort.py --species human --tissue CD4 --train_dataset 1013 1247 598 732 767 768 770 784 845 864 --test_dataset 315 340 376 381 390 404 437 490 551 559 --device cuda:2 --num_runs 5 > 1013-1247-598-732-767-768-770-784-845-864_315-340-376-381-390-404-437-490-551-559-scdeepsort-out.log 2>&1 &
python singlecellnet.py --species human --tissue CD4 --train_dataset 1013 1247 598 732 767 768 770 784 845 864 --test_dataset 315 340 376 381 390 404 437 490 551 559 --num_runs 5 > 1013-1247-598-732-767-768-770-784-845-864_315-340-376-381-390-404-437-490-551-559-singlecellnet-out.log 2>&1 &
python svm.py --species human --tissue CD4 --train_dataset 1013 1247 598 732 767 768 770 784 845 864 --test_dataset 315 340 376 381 390 404 437 490 551 559  --gpu 0 --num_runs 5 > 1013-1247-598-732-767-768-770-784-845-864_315-340-376-381-390-404-437-490-551-559-svm-out.log 2>&1 &



python main.py --species human --tissue Spleen --train_dataset 3043 3777 4029 4115 4362 4657  --test_dataset 1729 2125 2184 2724 2743 --count 800 --sweep_id dhi9mq68 --additional_sweep_ids w6hq93t7 >> 3043-3777-4029-4115-4362-4657_1729-2125-2184-2724-2743/out.log 2>&1 &


python main.py --species human --tissue Brain --train_dataset 328 --test_dataset 138 --count 800 --sweep_id x140d4n3 --additional_sweep_ids nnu92doo >> 328_138/out.log 2>&1 & #gpu is useless
python main.py --species human --tissue CD8 --train_dataset 1027 1357 1641 517 706 777 850 972  --test_dataset 245 332 377 398 405 455 470 492 --count 800 --gpu 4 --sweep_id 5ljj1bsw --additional_sweep_ids 1jcael3o >> 1027-1357-1641-517-706-777-850-972_245-332-377-398-405-455-470-492/out.log 2>&1 &
# python main.py --species human --tissue Immune --train_dataset 11407 1519 636 713 9054 9258 --test_dataset 1925 205 3323 6509 7572 --count 800 --gpu 2 --sweep_id 89551cea >> 11407-1519-636-713-9054-9258_1925-205-3323-6509-7572/out.log 2>&1 &
