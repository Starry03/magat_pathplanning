#!/bin/bash

source .venv/bin/activate
tensorboard --logdir=./tb_logs --port=6006 --bind_all

