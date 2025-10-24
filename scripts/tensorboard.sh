#!/bin/bash

source .venv/bin/activate
tensorboard --logdir=./Data/Tensorboard/PaperArchitecture_map20x20_rho1_10Agent --port=6006 --bind_all

