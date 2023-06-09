# Meta Temporal Point Processes
[[Paper](https://openreview.net/pdf?id=QZfdDpTX1uM)][[Poster](https://iclr.cc/media/PosterPDFs/ICLR%202023/11395.png?t=1682361273.0520558)][[OpenReview](https://openreview.net/forum?id=QZfdDpTX1uM)]

## Datasets
We provide the compressed datasets: Stack Overflow, Mooc, Reddit, Wiki, Sin, Uber, NYC Taxi, in this link.
Unzip the compressed file and locate it in the `$ROOT` directory.

## Setup
Setup the pipeline by installing dependencies using the following command.
pretrained models and utils.
```bash
pip install -r requirements.txt
```

## Pre-trained models
We also provide the checkpoints for Intensity free, THP+ and Attentive TPP on all the datasets.
Please download the compress file in this link, unzip it and locate it in the `$ROOT` directory.


## Train
A model can be trained using the following command.
```bash
python src/train.py data/datasets=$DATASET model=$MODEL
```
`$DATASET` can be chosen from `{so_fold1, mooc, reddit, wiki, sin, uber_drop, taxi_times_jan_feb}` and `$MODEL` can be chosen from `{intensity_free,thp_mix,attn_lnp}`.
Other configurations can be also easily modified using hydra syntax. Please refer to [hydra](https://hydra.cc/docs/intro/) for further details.

## Eval
A model can be evaluated on test datasets using the following command.
```bash
python src/eval.py data/datasets=$DATASET model=$MODEL
```
Here, the default checkpoint paths are set to the ones in `checkpoints` directory we provided above.
To use different checkpoints, please chagne `ckpt_path` argument in `configs/eval.yaml`.


## Citation
If you use this code or model for your research, please cite:

    @inproceedings{bae2023meta,
      title = {Meta Temporal Point Processes},
      author = {Bae, Wonho and Ahmed, Mohamed Osama and Tung, Frederick and Oliveira, Gabriel L},
      booktitle={The International Conference on Learning Representations (ICLR)},
      year={2023}
    }


## Acknowledgment
The pipeline is built on [PyTorch-Lightning Hydra Template](https://github.com/ashleve/lightning-hydra-template).
Intensity free is based on [the original implementation](https://github.com/shchur/ifl-tpp) and THP+ is based on (the corrected version of THP](https://github.com/yangalan123/anhp-andtt).



