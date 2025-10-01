#!/usr/bin/env bash

# Test script to verify CUDA multiprocessing fix
# This runs a minimal training iteration to check if the error is resolved

echo "Testing CUDA Multiprocessing Fix..."
echo "===================================="
echo ""

# Set environment for better error reporting
export CUDA_LAUNCH_BLOCKING=1

# Run with minimal test configuration
python main.py configs/dcp_OE_Random.json \
    --mode train \
    --map_density 1 \
    --map_w 20 \
    --num_agents 10 \
    --nGraphFilterTaps 2 \
    --trained_num_agents 10 \
    --commR 7 \
    --GSO_mode dist_GSO \
    --update_valid_set 1000 \
    --update_valid_set_epoch 70 \
    --threshold_SuccessRate 97 \
    --default_actionSelect \
    --guidance Project_G \
    --CNN_mode ResNetLarge_withMLP \
    --batch_numAgent \
    --test_num_processes 2 \
    --tb_ExpName CUDA_FIX_TEST \

echo ""
echo "===================================="
echo "Test completed!"
