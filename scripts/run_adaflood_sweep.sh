#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=64G
#SBATCH --time=3-0:00
#SBATCH --job-name=adaflood_sweep
#SBATCH --error=results/%x.%j.err
#SBATCH --output=results/%x.%j.out

source ~/pl/bin/activate

# Default values for arguments
seed=1
task=tpp
dataset=mooc
alpha=0.0
imb_factor=1.0

model=thp_mix_aux
scheduler=multistep

flood_levels=(null)
flood_level=0.0
lr=0.0001
weight_decay=0.00001
max_epochs=300

aux_d_model=64
tpp_aux_d_models=(64)
cls_aux_d_models=(64)
aux_lr=0.001
aux_weight_decay=0.001

affine_train=null
aux_num=2

#gammas=(0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95)
#gammas=(0.2 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95) # 0.50 0.60 0.70 0.80 0.90 1.0)
tpp_gammas=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9) # 0.50 0.60 0.70 0.80 0.90 1.0)
#tpp_gammas=(0.0) # 0.50 0.60 0.70 0.80 0.90 1.0)
#tpp_gammas=(0.05 0.15 0.25 0.35) # 0.50 0.60 0.70 0.80 0.90 1.0)
#cls_gammas=(0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95) # 0.50 0.60 0.70 0.80 0.90 1.0)
#cls_gammas=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9) # 0.50 0.60 0.70 0.80 0.90 1.0)
cls_gammas=(0.5 0.6 0.7 0.8 0.9) # 0.50 0.60 0.70 0.80 0.90 1.0)
#cls_gammas=(0.5) # 0.50 0.60 0.70 0.80 0.90 1.0)



# Parsing arguments
while getopts ":d:f:m:l:w:k:s:i:t:e:p:j:a:x:y:b:" flag; do
  case "${flag}" in
    s) seed=${OPTARG};;
    t) task=${OPTARG};;
    d) dataset=${OPTARG};;
    a) alpha=${OPTARG};;
    e) scheduler=${OPTARG};;
    f) flood_level=${OPTARG};;
    m) model=${OPTARG};;
    w) weight_decay=${OPTARG};;
    l) lr=${OPTARG};;
    k) aux_d_model=${OPTARG};;
    x) aux_lr=${OPTARG};;
    y) aux_weight_decay=${OPTARG};;
    i) affine_train=${OPTARG};;
    p) max_epochs=${OPTARG};;
    j) aux_num=${OPTARG};;
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
    if [ $model == "thp_mix_aux" ]
    then
        experiment=adaflood_tpp
    elif [ $model == "intensity_free_aux" ]
    then
        experiment=adaflood_if
    fi
    #if [ $dataset == "uber_drop" ] || [ $dataset == "taxi_times_jan_feb" ]
    #then
    #    gammas=(0.0)
    #fi
    #for aux_d_model in ${tpp_aux_d_models[@]}; do
    #for flood_level in {null,-10.0,-1.0,0.0,1.0,10.0}; do
    for gamma in ${tpp_gammas[@]}; do
        echo "**************** Script Arguments **************"
        echo "seed: $seed";
        echo "task: $task";
        echo "dataset: $dataset";
        echo "alpha: $alpha";
        echo "model: $model";
        echo "lr: $lr";
        echo "weight_decay: $weight_decay";
        echo "aux_num: $aux_num";
        echo "aux_d_model: $aux_d_model";
        echo "aux_lr: $aux_lr";
        echo "aux_weight_decay: $aux_weight_decay";
        echo "affine_train: $affine_train";
        echo "gamma: $gamma";
        echo "************************************************"
        python src/train_tpp.py seed=$seed experiment=$experiment trainer.max_epochs=$max_epochs \
            data/datasets=$dataset data.alpha=$alpha model=$model \
            model.optimizer.lr=$lr model.optimizer.weight_decay=$weight_decay tags=["adaflood","final"] \
            model.net.aux_lr=$aux_lr model.net.aux_weight_decay=$aux_weight_decay model.net.aux_d_model=$aux_d_model \
            model.criterion.affine_train=$affine_train data.aux_num=$aux_num model.criterion.gamma=$gamma
    done
elif [ $task == "cls" ]
then
    if [ $dataset == "cars" ]
    then
        experiment=adaflood_cls_large
    else
        experiment=adaflood_cls
    fi
    for aux_d_model in ${cls_aux_d_models[@]}; do
        #for gamma in {0.3,0.4,0.5,0.6,0.7}; do
        #for gamma in {0.35,0.45,0.55,0.65,0.75}; do
        #for gamma in {0.40,0.41,0.42,0.43,0.44}; do
        #for gamma in {0.30,0.35,0.4,0.45}; do
        #for gamma in {0.65,0.75}; do
        #for gamma in {0.1,0.2,0.3,0.4,0.5}; do
        #for gamma in {0.05,0.15,0.25,0.35,0.45}; do
        #for gamma in {0.80,0.85,0.90,0.95,0.1,0.2}; do
        for gamma in ${cls_gammas[@]}; do
        #for gamma in {0.05,0.15,0.25}; do
        #for flood_level in {0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75}; do
    #        for flood_level in ${tmp_fl[@]}; do # (0.3, 0.4, 0.5, 0.6)
            echo "**************** Script Arguments **************"
            echo "seed: $seed";
            echo "task: $task";
            echo "dataset: $dataset";
            echo "alpha: $alpha";
            echo "imb_factor: $imb_factor";
            echo "model: $model";
            echo "lr: $lr";
            echo "weight_decay: $weight_decay";
            echo "aux_num: $aux_num";
            echo "aux_d_model: $aux_d_model";
            echo "aux_lr: $aux_lr";
            echo "aux_weight_decay: $aux_weight_decay";
            echo "affine_train: $affine_train";
            echo "gamma: $gamma";
            echo "************************************************"
            python src/train_cls.py seed=$seed experiment=$experiment trainer.max_epochs=$max_epochs \
                data/datasets=$dataset data.alpha=$alpha data.imb_factor=$imb_factor model=$model \
                model.optimizer.lr=$lr model.optimizer.weight_decay=$weight_decay \
                model/scheduler=$scheduler tags=["adaflood","final_test"] \
                model.net.aux_lr=$aux_lr model.net.aux_weight_decay=$aux_weight_decay model.net.aux_d_model=$aux_d_model \
                model.criterion.affine_train=$affine_train model.criterion.gamma=$gamma data.aux_num=$aux_num
        #    done
        done
    done
fi


#for aux_d_model in {16,32,64,128}; do
#    echo "**************** Script Arguments **************"
#    echo "seed: $seed";
#    echo "task: $task";
#    echo "dataset: $dataset";
#    echo "model: $model";
#    echo "lr: $lr";
#    echo "weight_decay: $weight_decay";
#    echo "aux_d_model: $aux_d_model";
#    echo "aux_lr: $aux_lr";
#    echo "aux_weight_decay: $aux_weight_decay";
#    echo "affine_trainable: $affine_trainable";
#    echo "************************************************"
#
#    python src/train_tpp.py seed=$seed experiment=adaflood_tpp data/datasets=$dataset model=$model \
#        model.optimizer.lr=$lr model.optimizer.weight_decay=$weight_decay tags=["adaflood","fixed"] \
#        model.net.aux_lr=$aux_lr model.net.aux_weight_decay=$aux_weight_decay model.net.aux_d_model=$aux_d_model \
#        model.criterion.affine_trainable=$affine_trainable
#done

