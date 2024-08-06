# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#!/bin/bash

# Default values for arguments
seed=1
task=tpp
dataset=uber_drop
alpha=0.0
imb_factor=1.0

model=thp_mix
scheduler=multistep
flood_level=0.0
lr=0.0001
weight_decay=0.00001
max_epochs=300
aux_num=0


# Parsing arguments
while getopts ":d:f:m:l:w:s:t:e:p:a:b:j:" flag; do
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
    a) alpha=${OPTARG};;
    b) imb_factor=${OPTARG};;
    j) aux_num=${OPTARG};;
    :)                                         # If expected argument omitted:
      echo "Error: -${OPTARG} requires an argument."
      exit_abnormal;;                          # Exit abnormally.
    *)                                         # If unknown (any other) option:
      exit_abnormal;;                          # Exit abnormally.
  esac
done


echo "**************** Script Arguments **************"
echo "seed: $seed";
echo "task: $task";
echo "dataset: $dataset";
echo "alpha: $alpha";
echo "imb_factor: $imb_factor";
echo "model: $model";
echo "lr: $lr";
echo "weight_decay: $weight_decay";
echo "************************************************"

if [ $task == "tpp" ]
then
    if [ $model == "thp_mix" ]
    then
        experiment=tpp
    elif [ $model == "intensity_free" ]
    then
        experiment=if
    fi
    python src/train_tpp.py seed=$seed experiment=$experiment trainer.max_epochs=$max_epochs \
        data/datasets=$dataset data.alpha=$alpha model=$model \
        model.optimizer.lr=$lr model.optimizer.weight_decay=$weight_decay data.aux_num=$aux_num tags=["tpp","final"]
elif [ $task == "cls" ]
then
    experiment=cls
    python src/train_cls.py seed=$seed experiment=$experiment trainer.max_epochs=$max_epochs \
        data/datasets=$dataset data.alpha=$alpha data.imb_factor=$imb_factor model=$model \
        model.optimizer.lr=$lr model.optimizer.weight_decay=$weight_decay \
        model/scheduler=$scheduler data.aux_num=$aux_num tags=["cls","test"]
fi


