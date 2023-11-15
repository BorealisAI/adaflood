#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=64G
#SBATCH --time=3-0:00
#SBATCH --job-name=base_sweep
#SBATCH --error=results/%x.%j.err
#SBATCH --output=results/%x.%j.out

source ~/pl/bin/activate

# Default values for arguments
seed=1
task=tpp
dataset=mooc
alpha=0.0
imb_factor=1.0

model=thp_mix
scheduler=multistep
flood_level=0.0
lr=0.0001
weight_decay=0.00001
max_epochs=300
aux_num=0

ckpt_epoch=null

# Parsing arguments
while getopts ":d:f:m:l:w:s:t:e:p:a:b:j:c:" flag; do
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
    c) ckpt_epoch=${OPTARG};;
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
    python src/eval_tpp.py seed=$seed experiment=$experiment trainer.max_epochs=$max_epochs ckpt_epoch=$ckpt_epoch \
        data/datasets=$dataset data.alpha=$alpha model=$model \
        model.optimizer.lr=$lr model.optimizer.weight_decay=$weight_decay data.aux_num=$aux_num tags=["tpp","final"]
elif [ $task == "cls" ]
then
    python src/eval_cls.py seed=$seed experiment=cls trainer.max_epochs=$max_epochs ckpt_epoch=$ckpt_epoch \
        data/datasets=$dataset data.alpha=$alpha data.imb_factor=$imb_factor model=$model \
        model.optimizer.lr=$lr model.optimizer.weight_decay=$weight_decay \
        model/scheduler=$scheduler data.aux_num=$aux_num tags=["cls","final"]
fi


