# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#!/bin/bash

# Default values for arguments
seed=1
task=tpp
dataset=mooc
alpha=0.0
imb_factor=1.0

model=thp_mix
criterion=flood
scheduler=multistep

flood_level=0.0
lr=0.001
weight_decay=0.001
use_weight_decay=false

aux_d_model=64
aux_lr=0.001
aux_weight_decay=0.001
aux_num=0

affine_train=null
compute_node=false

# abghopquvz
# Parsing arguments
while getopts ":d:f:m:l:w:r:k:s:i:t:n:e:j:a:x:y:b:u:" flag; do
  case "${flag}" in
    s) seed=${OPTARG};;
    t) task=${OPTARG};;
    d) dataset=${OPTARG};;
    f) flood_level=${OPTARG};;
    m) model=${OPTARG};;
    w) weight_decay=${OPTARG};;
    l) lr=${OPTARG};;
    r) criterion=${OPTARG};;
    e) scheduler=${OPTARG};;
    k) aux_d_model=${OPTARG};;
    x) aux_lr=${OPTARG};;
    y) aux_weight_decay=${OPTARG};;
    i) affine_train=${OPTARG};;
    n) compute_node=${OPTARG};;
    j) aux_num=${OPTARG};;
    a) alpha=${OPTARG};;
    b) imb_factor=${OPTARG};;
    u) use_weight_decay=${OPTARG};;
    :)                                         # If expected argument omitted:
      echo "Error: -${OPTARG} requires an argument."
      exit_abnormal;;                          # Exit abnormally.
    *)                                         # If unknown (any other) option:
      exit_abnormal;;                          # Exit abnormally.
  esac
done

tmp_weight_decay=$weight_decay

# determine a node - compute or interactive
if [ $compute_node == "true" ]
then
    command="sbatch"
else
    command="bash"
fi

# set lrs and weight decays
if [ $task == "tpp" ]
then
    if [ $dataset == "uber_drop" ]
    then
        if [ $model == "thp_mix" ] || [ $model == "thp_mix_aux" ]
        then
            lrs=(0.001)
            weight_decays=(0.001)
            lr=0.001
            weight_decay=0.001
        elif [ $model == "intensity_free" ]  || [ $model == "intensity_free_aux" ]
        then
            lrs=(0.0001)
            weight_decays=(0.01)
            lr=0.0001
            weight_decay=0.01
        fi
    else
        lrs=(0.01 0.001 0.0001)
        weight_decays=(0.01 0.001 0.0001 0.00001)
    fi
    max_epochs=2000
elif [ $task == "cls" ]
then
    if [ $dataset == "cifar100" ]
    then
        lrs=(0.1)
        weight_decays=(0.0)
        scheduler=multistep2
        max_epochs=300
        lr=0.1
        weight_decay=0.0
    else
        lrs=(0.01)
        weight_decays=(0.01)
        scheduler=multistep
        max_epochs=300
    fi
fi

if [ $use_weight_decay == "true" ]
then
    weight_decays=($tmp_weight_decay)
    weight_decay=$tmp_weight_decay
fi


if [ $criterion == "aux" ]
then
    d_models=(64)
    for d_model in ${d_models[@]}; do
        $command scripts/run_aux_sweep.sh -s $seed -t $task -p $max_epochs -d $dataset -a $alpha -b $imb_factor \
            -m $model -l $lr -w $weight_decay -e $scheduler -z $d_model -j $aux_num
    done
else
    for lr in ${lrs[@]}; do
        for weight_decay in ${weight_decays[@]}; do
            echo "************************************************"
            echo "criterion: $criterion"
            echo "task: $task"
            echo "lr: $lr";
            echo "weight_decay: $weight_decay";

            if [ $criterion == "base" ]
            then
                $command scripts/run_base.sh -s $seed -t $task -p $max_epochs -d $dataset -a $alpha -b $imb_factor \
                    -m $model -l $lr -w $weight_decay -e $scheduler -j $aux_num
            elif [ $criterion == "flood" ]
            then
                $command scripts/run_flood_sweep.sh -s $seed -t $task -p $max_epochs -d $dataset -a $alpha -b $imb_factor \
                    -m $model -l $lr -w $weight_decay -e $scheduler
            elif [ $criterion == "iflood" ]
            then
                $command scripts/run_iflood_sweep.sh -s $seed -t $task -p $max_epochs -d $dataset -a $alpha -b $imb_factor \
                    -m $model -l $lr -w $weight_decay -e $scheduler
            elif [ $criterion == "adaflood" ]
            then
                $command scripts/run_adaflood_sweep.sh -s $seed -t $task -p $max_epochs -d $dataset -a $alpha -b $imb_factor \
                    -m $model -l $lr -w $weight_decay -e $scheduler -x $aux_lr -y $aux_weight_decay -i $affine_train -j $aux_num
            fi
        done
    done
fi

