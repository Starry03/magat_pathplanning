#!/bin/bash

source ../.venv/bin/activate

MAP_FOLDER="./maps/"
MAP_SET="./maps/map20x20_density_p1"
SOLUTIONS_FOLDER="./solutions/"
TRAINSET_FOLDER="./trainset/"

MAP_SIZE=20
NUM_AGENTS=10
MAP_COMPLEXITY=0.005
MAP_DENSITY=0.1

NUM_DATASET=30000
DIV_TRAIN=24000
DIV_VALID=3000
DIV_TEST=3000

# python Generate_cropmap.py  --random_map --gen_CasePool --gen_map_type \
#     random --chosen_solver ECBS --map_width ${MAP_SIZE} --map_density ${MAP_DENSITY}  \
#     --map_complexity ${MAP_COMPLEXITY} --num_agents ${NUM_AGENTS} --num_dataset ${NUM_DATASET} \
#     --num_caseSetup_pEnv 50 --path_save ${MAP_FOLDER}

# python CasesSolver_cropfromMap_fixedLength.py --loadmap_TYPE \
#     random --random_map --gen_CasePool --chosen_solver ECBS --map_width ${MAP_SIZE} \
#     --map_density ${MAP_DENSITY}  --map_complexity ${MAP_COMPLEXITY} --num_agents ${NUM_AGENTS} \
#     --num_dataset ${NUM_DATASET} --num_caseSetup_pEnv 50 \
#     --path_save ${SOLUTIONS_FOLDER} --path_loadSourceMap ${MAP_SET}

python DataGen_Transformer_split_IDMap.py  --num_agents ${NUM_AGENTS} --map_w ${MAP_SIZE} \
    --map_density ${MAP_DENSITY}   --div_train ${DIV_TRAIN} --div_valid ${DIV_VALID} --div_test ${DIV_TEST} \
    --div_train_IDMap 0 --div_test_IDMap 160 --div_valid_IDMap 180 --maxNum_Map 200 \
    --solCases_dir ${SOLUTIONS_FOLDER}   --dir_SaveData ${TRAINSET_FOLDER}  \
    --guidance Project_G
