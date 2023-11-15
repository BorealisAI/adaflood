#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=64G
#SBATCH --time=3-0:00
#SBATCH --job-name=aux
#SBATCH --error=logs/%x.%j.err
#SBATCH --output=logs/%x.%j.out

source ~/pl/bin/activate

# Default values for arguments
seed=1
dataset=mooc
model=thp_mix
lr=0.0001
weight_decay=0.00001
batch_size=16
d_model=64

# Parsing arguments
while getopts ":d:f:m:l:w:b:k:s:" flag; do
  case "${flag}" in
    s) seed=${OPTARG};;
    d) dataset=${OPTARG};;
    f) flood_level=${OPTARG};;
    m) model=${OPTARG};;
    w) weight_decay=${OPTARG};;
    l) lr=${OPTARG};;
    b) batch_size=${OPTARG};;
    k) d_model=${OPTARG};;
    :)                                         # If expected argument omitted:
      echo "Error: -${OPTARG} requires an argument."
      exit_abnormal;;                          # Exit abnormally.
    *)                                         # If unknown (any other) option:
      exit_abnormal;;                          # Exit abnormally.
  esac
done


for aux in {aux1,aux2}; do
    echo "**************** Script Arguments **************"
    echo "seed: $seed";
    echo "dataset: $dataset";
    echo "model: $model";
    echo "aux: $aux"
    echo "lr: $lr";
    echo "weight_decay: $weight_decay";
    echo "************************************************"

    python src/train_tpp.py seed=$seed experiment=${aux}_tpp data/datasets=$dataset model=$model \
        model.optimizer.lr=$lr model.optimizer.weight_decay=$weight_decay \
        model.net.d_model=$d_model
done

