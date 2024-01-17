#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --mem=160G
#SBATCH --cpus-per-task=32
#SBATCH --time=5-0:00
#SBATCH --job-name=aux
#SBATCH --error=results/%x.%j.err
#SBATCH --output=results/%x.%j.out

source ~/pl/bin/activate

# Default values for arguments
seed=1
task=tpp
dataset=mooc
alpha=0.0
imb_factor=1.0

scheduler=multistep
model=thp_mix
lr=0.0001
weight_decay=0.00001
batch_size=16
max_epochs=300
d_model=64
aux_num=10

ckpt_path=null
#ckpt_path=/home/whbae/meta-tpp-lightning/results/imagenet100_resnet34_alpha0.0_imb1.0_cls_aux-2_test/seed1/lr0.1_wd0.0001_mdim64/aux-1/checkpoints/epoch_0190.ckpt
#ckpt_path=/home/whbae/meta-tpp-lightning/results/imagenet100_resnet34_alpha0.3_imb1.0_cls_aux-2_test/seed1/lr0.1_wd0.0001_mdim64/aux-1/checkpoints/epoch_0050.ckpt
#ckpt_path=/home/whbae/meta-tpp-lightning/results/imagenet100_resnet34_alpha0.6_imb1.0_cls_aux-2_test/seed1/lr0.1_wd0.0001_mdim64/aux-1/checkpoints/epoch_0050.ckpt
#ckpt_path=/home/whbae/meta-tpp-lightning/results/cars_resnet34_alpha0.0_imb1.0_cls_final/seed1/lr0.1_wd0.0005/checkpoints/epoch_0249.ckpt
#ckpt_path=/home/whbae/meta-tpp-lightning/results/cars_resnet34_alpha0.6_imb1.0_cls_final/seed1/lr0.1_wd0.0005/checkpoints/epoch_0130.ckpt
#ckpt_path=/home/whbae/meta-tpp-lightning/results/cars_resnet34_alpha0.2_imb1.0_cls_final/seed1/lr0.1_wd0.0005/checkpoints/epoch_0287.ckpt
#ckpt_path=/home/whbae/meta-tpp-lightning/results/animal_resnet18_alpha0.3_imb1.0_cls_final/seed1/lr0.1_wd0.0001/checkpoints/epoch_0097.ckpt
ckpt_path=/home/whbae/meta-tpp-lightning/results/food101_resnet34_alpha0.3_imb1.0_cls_final/seed1/lr0.1_wd0.0001/checkpoints/epoch_0082.ckpt

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
    if [ $dataset == "cars" ]
    then
        experiment=aux_cls_large
    else
        experiment=aux_cls
    fi
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


#if [ $task == "tpp" ]
#then
#    for d_model in {16,32,64,128}; do
#        for aux in {aux1,aux2}; do
#            echo "**************** Script Arguments **************"
#            echo "seed: $seed";
#            echo "task: $task";
#            echo "dataset: $dataset";
#            echo "model: $model";
#            echo "d_model: $d_model";
#            echo "aux: $aux"
#            echo "lr: $lr";
#            echo "weight_decay: $weight_decay";
#            echo "************************************************"
#            python src/train_tpp.py seed=$seed experiment=${aux}_tpp data/datasets=$dataset model=$model \
#                model.optimizer.lr=$lr model.optimizer.weight_decay=$weight_decay model.net.d_model=$d_model
#        done
#    done
#elif [ $task == "cls" ]
#then
#    for d_model in {16,32,64,128}; do
#        for aux in {aux1,aux2}; do
#            echo "**************** Script Arguments **************"
#            echo "seed: $seed";
#            echo "task: $task";
#            echo "dataset: $dataset";
#            echo "model: $model";
#            echo "d_model: $d_model";
#            echo "aux: $aux"
#            echo "lr: $lr";
#            echo "weight_decay: $weight_decay";
#            echo "************************************************"
#            python src/train_cls.py seed=$seed experiment=${aux}_cls data/datasets=$dataset model=$model \
#                model.optimizer.lr=$lr model.optimizer.weight_decay=$weight_decay model.net.d_model=$d_model
#        done
#    done
#fi


