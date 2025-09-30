#!/bin/bash

source ~/Documents/uni_project/magat_pathplanning/.venv/bin/activate

# python main.py ~/Documents/uni_project/magat_pathplanning/configs/test_train.json --mode train --test_epoch 240 \
#     --test_general --log_time_trained 1602191336   --nGraphFilterTaps 2 --nAttentionHeads 1 \
#     --trained_num_agents 10 --trained_map_w 20   --commR 7  --list_map_w 20 \
#     --list_agents 10   --list_num_testset 100    --GSO_mode dist_GSO  \
#     --action_select exp_multinorm  --guidance Project_G --CNN_mode Default \
#     --batch_numAgent --test_num_processes 2   --tb_ExpName GNN_Resnet_3Block_distGSO_baseline_128

python main.py ~/Documents/uni_project/magat_pathplanning/configs/test_train.json --mode train --map_density 1 --map_w 20 \
    --nGraphFilterTaps 2  --num_agents 10  --trained_num_agents 10  --commR 7  \
    --load_num_validset 1000 --update_valid_set 1000 --update_valid_set_epoch 100 \
    --GSO_mode dist_GSO --default_actionSelect \
    --guidance Project_G --CNN_mode ResNetLarge_withMLP  --batch_numAgent \
    --test_num_processes 2  --nAttentionHeads 1 --attentionMode KeyQuery  \
    --tb_ExpName DotProduct_GAT_Resnet_3Block_MLP128_distGSO_baseline_128_Validset_1K_RandomMap
