#!/usr/bin/env bash

#use this line to run the main.py file with a specified config file
#python3 main.py PATH_OF_THE_CONFIG_FILE

#export CUDA_DEVICE_ORDER=PCI_BUS_ID
#export CUDA_VISIBLE_DEVICES= 0, 1, 2

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"


#####################################################################
#                                                                   #
#                  Control Group 40 - CG 40                         #
#               (relative coordination + DMAP )                     #
#                                                                   #
#####################################################################

# ##################################
# #########
# #########    Training  generation
# #########
# ##################################

source .venv/bin/activate

python main.py configs/dcp_ECBS.json --mode train \
    --map_density 1 --map_w 20 --num_agents 10  --nGraphFilterTaps 2  \
    --trained_num_agents 10 --commR 7 --GSO_mode dist_GSO --update_valid_set 1000 \
    --update_valid_set_epoch 70 --threshold_SuccessRate 97 --default_actionSelect \
    --guidance Project_G  --batch_numAgent --test_num_processes 0
    # --tb_ExpName GNN_Resnet_3Block_distGSO_baseline_128 \
    # --CNN_mode ResNetLarge_withMLP