#!/bin/sh

exp_name=" baseline"
python train.py --exp_name $exp_name --dn PS --v1 0.25 --v2 0.1  --num_epoch 18 --save
echo "----------------- Start Stage 2 -----------------"
python evaluate.py --exp_name $exp_name --eval --dn PS

