#!/bin/bash

source ~/Documents/uni_project/magat_pathplanning/.venv/bin/activate

python main.py ~/Documents/uni_project/magat_pathplanning/configs/test_train.json --mode test --test_epoch 240 \
    --test_general --log_time_trained 1602191336   --nGraphFilterTaps 2 --nAttentionHeads 1 \
    --trained_num_agents 10 --trained_map_w 20   --commR 7  --list_map_w 20 \
    --list_agents 10   --list_num_testset 100    --GSO_mode dist_GSO  \
    --action_select exp_multinorm  --guidance Project_G --CNN_mode Default \
    --batch_numAgent --test_num_processes 2   --tb_ExpName GNN_Resnet_3Block_distGSO_baseline_128
