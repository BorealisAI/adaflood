# Adaflood
[[Paper](https://openreview.net/pdf?id=2s5YU6CSEz)][[OpenReview](https://openreview.net/forum?id=2s5YU6CSEz)]

## Setup
Setup the pipeline by installing dependencies using the following command.
pretrained models and utils.
```bash
pip install -r requirements.txt
```

## Train
A model can be trained (with hyper-parameter search) using the following command.
```bash
bash scripts/run_sweep.sh -d $DATASET -m $MODEL -r $CRITERION
```
`$DATASET` can be chosen from `{uber_drop, cifar100}`, `$MODEL` can be chosen from `{intensity_free,thp_mix,resnet18,}`.
Also, `$CRITERION` can be chosen from `{cls,flood,iflood,aux,adaflood}` for classification and `{tpp,flood,iflood,aux,adaflood}` for TPP tasks.
Other configurations can be also easily modified using hydra syntax. Please refer to `scripts/run_sweep.sh` and [hydra](https://hydra.cc/docs/intro/) for further details.


## Citation
If you use this code or model for your research, please cite:

    @article{bae2023adaflood,
      title={AdaFlood: Adaptive Flood Regularization},
      author={Bae, Wonho and Ren, Yi and Ahmed, Mohamad Osama and Tung, Frederick and Sutherland, Danica J and Oliveira, Gabriel L},
      journal={Transactions on Machine Learning Research (TMLR)},
      year={2023}
    }


## Acknowledgment
The pipeline is built on [PyTorch-Lightning Hydra Template](https://github.com/ashleve/lightning-hydra-template).
Intensity free is based on [the original implementation](https://github.com/shchur/ifl-tpp) and THP+ is based on (Meta TPP](https://github.com/BorealisAI/meta-tpp).



