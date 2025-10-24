source .venv/bin/activate

# get current time
now=$(date +"%Y%m%d_%H%M%S")
echo "Current time: $now"

## test

python ./test/main.py /home/starry/Documents/uni_project/magat_pathplanning/configs/test/test_train_pl.json \
    --mode test \
    --map_density 1 --map_w 20 --num_agents 10 --nGraphFilterTaps 3   \
    --trained_num_agents 10 \
    --commR 5 \
    --GSO_mode full_GSO \
    --update_valid_set 1000 \
    --update_valid_set_epoch 70 --threshold_SuccessRate 90 --default_actionSelect \
    --guidance Project_G \
    --test_num_processes 0 \
    --tb_ExpName PaperArchitecture_TestRun_${now} \
