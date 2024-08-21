#!/bin/bash

# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# Default values for arguments
seed=1
task=tpp
dataset=uber_drop
alpha=0.0
imb_factor=1.0

scheduler=multistep
model=thp_mix
lr=0.0001
weight_decay=0.00001
max_epochs=300
d_model=64
aux_num=10

ckpt_path=null

# Parsing arguments
while getopts ":d:f:m:l:w:s:t:e:p:z:n:j:a:b:" flag; do
  case "${flag}" in
    s) seed=${OPTARG};;
    t) task=${OPTARG};;
    d) dataset=${OPTARG};;
    e) scheduler=${OPTARG};;
    f) flood_level=${OPTARG};;
    m) model=${OPTARG};;
    w) weight_decay=${OPTARG};;
    l) lr=${OPTARG};;
    p) max_epochs=${OPTARG};;
    z) d_model=${OPTARG};;
    j) aux_num=${OPTARG};;
    a) alpha=${OPTARG};;
    b) imb_factor=${OPTARG};;
    :)                                         # If expected argument omitted:
      echo "Error: -${OPTARG} requires an argument."
      exit_abnormal;;                          # Exit abnormally.
    *)                                         # If unknown (any other) option:
      exit_abnormal;;                          # Exit abnormally.
  esac
done


if [ $task == "tpp" ]
then
    if [ $model == "thp_mix" ]
    then
        experiment=aux_tpp
    elif [ $model == "intensity_free" ]
    then
        experiment=aux_if
    fi
    for (( aux_idx=0; aux_idx<$aux_num; aux_idx+=1 )); do
        echo "**************** Script Arguments **************"
        echo "seed: $seed";
        echo "task: $task";
        echo "dataset: $dataset";
        echo "alpha: $alpha";
        echo "model: $model";
        echo "d_model: $d_model";
        echo "lr: $lr";
        echo "weight_decay: $weight_decay";
        echo "************************************************"
        python src/train_tpp.py seed=$seed experiment=$experiment trainer.max_epochs=$max_epochs \
            data/datasets=$dataset data.alpha=$alpha model=$model \
            model.optimizer.lr=$lr model.optimizer.weight_decay=$weight_decay model.net.d_model=$d_model \
            data.aux_idx=$aux_idx data.aux_num=$aux_num
    done
elif [ $task == "cls" ]
then
    experiment=aux_cls
    if (( $aux_num <= 0 ))
    then
        python src/train_cls.py seed=$seed experiment=$experiment trainer.max_epochs=$max_epochs \
                data/datasets=$dataset data.alpha=$alpha data.imb_factor=$imb_factor model=$model \
                model.optimizer.lr=$lr model.optimizer.weight_decay=$weight_decay model/scheduler=$scheduler model.net.d_model=$d_model \
                data.aux_idx=-1 data.aux_num=$aux_num ckpt_path=$ckpt_path
    else
        for (( aux_idx=0; aux_idx<$aux_num; aux_idx+=1 )); do
            echo "**************** Script Arguments **************"
            echo "seed: $seed";
            echo "task: $task";
            echo "dataset: $dataset";
            echo "alpha: $alpha";
            echo "imb_factor: $imb_factor";
            echo "model: $model";
            echo "d_model: $d_model";
            echo "lr: $lr";
            echo "weight_decay: $weight_decay";
            echo "aux_idx / aux_num: $aux_idx / $aux_num";
            echo "************************************************"
            python src/train_cls.py seed=$seed experiment=$experiment trainer.max_epochs=$max_epochs \
                data/datasets=$dataset data.alpha=$alpha data.imb_factor=$imb_factor model=$model \
                model.optimizer.lr=$lr model.optimizer.weight_decay=$weight_decay model/scheduler=$scheduler model.net.d_model=$d_model \
                data.aux_idx=$aux_idx data.aux_num=$aux_num
        done
    fi
fi
