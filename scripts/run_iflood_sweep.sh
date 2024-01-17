#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --mem=160G
#SBATCH --cpus-per-task=32
#SBATCH --time=3-0:00
#SBATCH --job-name=iflood_sweep
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


# Parsing arguments
while getopts ":d:f:m:l:w:s:t:e:p:a:b:" flag; do
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
        experiment=iflood_tpp
    elif [ $model == "intensity_free" ]
    then
        experiment=iflood_if
    fi
    #for flood_level in {-10.0,-3.0,-1.0,0.0,1.0,3.0}; do
    #for flood_level in {-100.0,-30.0,10.0,30.0,100.0}; do
    #for flood_level in {-30.0,-10.0,-1.0,0.0,1.0,10.0}; do
    #for flood_level in {-50.0,-45.0,-40.0,-35.0,-30.0,-25.0,-20.0,-15.0,-5.0}; do
    for flood_level in {-2.0,-3.0,-4.0,2.0,3.0,4.0}; do
    #for flood_level in {2.0,3.0,4.0}; do
        echo "**************** Script Arguments **************"
        echo "seed: $seed";
        echo "task: $task";
        echo "dataset: $dataset";
        echo "alpha: $alpha";
        echo "model: $model";
        echo "lr: $lr";
        echo "weight_decay: $weight_decay";
        echo "flood_level: $flood_level";
        echo "************************************************"
        python src/train_tpp.py seed=$seed experiment=$experiment trainer.max_epochs=$max_epochs \
            data/datasets=$dataset data.alpha=$alpha model=$model \
            model.optimizer.lr=$lr model.optimizer.weight_decay=$weight_decay \
            model.criterion.flood_level=$flood_level tags=["iflood","final"]
    done
elif [ $task == "cls" ]
then
    if [ $dataset == "cars" ]
    then
        experiment=iflood_cls_large
    else
        experiment=iflood_cls
    fi
    #for flood_level in {0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5}; do
    #for flood_level in {0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0}; do
    #for flood_level in {0.05,0.15,0.25,0.3,0.35,0.4,0.45,0.5}; do
    #for flood_level in {0.3,0.4,0.5,0.6}; do
    #for flood_level in {0.35,0.45,0.55,0.65}; do
    #for flood_level in {0.3,0.4,0.5,0.6,0.7}; do
    #for flood_level in {0.35,0.45,0.55,0.65,0.75}; do
    #for flood_level in {0.3,0.4,0.5,0.6,0.7}; do
    #for flood_level in {0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75}; do
    #for flood_level in {0.2,0.25,0.3,0.35,0.4,0.45,0.5,55,0.6,0.65,0.7}; do
    #for flood_level in {0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.80,0.85,0.90,0.95}; do
    for flood_level in {0.05,0.1,0.15,0.2,0.25,0.3}; do
    #for flood_level in {0.01,0.02,0.03,0.04}; do
    #for flood_level in {0.05,0.03}; do
        echo "**************** Script Arguments **************"
        echo "seed: $seed";
        echo "task: $task";
        echo "dataset: $dataset";
        echo "alpha: $alpha";
        echo "imb_factor: $imb_factor";
        echo "model: $model";
        echo "lr: $lr";
        echo "weight_decay: $weight_decay";
        echo "flood_level: $flood_level";
        echo "************************************************"
        python src/train_cls.py seed=$seed experiment=$experiment trainer.max_epochs=$max_epochs \
            data/datasets=$dataset data.alpha=$alpha data.imb_factor=$imb_factor model=$model \
            model.optimizer.lr=$lr model.optimizer.weight_decay=$weight_decay \
            model.criterion.flood_level=$flood_level model/scheduler=$scheduler \
            tags=["iflood","final"]
    done
fi



#for flood_level in {-100.0,-10.0,-3.0,-1.0,0.0,1.0}; do
##for flood_level in {-1000.0,-100.0}; do
#    echo "**************** Script Arguments **************"
#    echo "seed: $seed";
#    echo "task: $task";
#    echo "dataset: $dataset";
#    echo "model: $model";
#    echo "lr: $lr";
#    echo "weight_decay: $weight_decay";
#    echo "flood_level: $flood_level";
#    echo "************************************************"
#
#    python src/train_tpp.py seed=$seed experiment=iflood_tpp data/datasets=$dataset model=$model \
#        model.optimizer.lr=$lr model.optimizer.weight_decay=$weight_decay \
#        model.criterion.flood_level=$flood_level tags=["iflood","fixed"]
#done

