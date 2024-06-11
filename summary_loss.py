import os
import itertools
import argparse
import numpy as np
import pandas as pd

log_dir = os.path.join(f"{os.getcwd()}", "logs")


def format_float(num):
    if num == 0.0:
        return str(0.0)
    return np.format_float_positional(num, trim='-')

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", required=True,
                        choices=["so_fold1", "mooc", "reddit", "wiki",
                                 "sin", "sin_long", "uber", "taxi"])
    parser.add_argument("-m", "--model", required=True,
                        choices=["thp_mix", "if"])
    parser.add_argument("-t", "--train_tag", default="")
    parser.add_argument("-e", "--eval_tag", default="test")
    parser.add_argument("-s", "--seed", default=1)
    parser.add_argument("-k", "--dim", default=16)

    parser.add_argument("-g", "--diffuser", default="tddpm", choices=["ddpm", "tddpm"])

    #parser.add_argument("-i", "--n_layer", default=8)
    #parser.add_argument("-h", "--n_head", default=64)
    #parser.add_argument("-p", "--multiplier", default=4)
    #parser.add_argument("-x", "--diff_lr", default=0.0002)
    #parser.add_argument("-y", "--diff_weight_decay", default=0.0)

    #parser.add_argument("-e", "--eval_metric", default="val/loss",
    #                    choices=["val/loss"])
    args = parser.parse_args()

    eval_metric = "val/loss"

    # Determine lr and weight_decay for TPP
    if args.data == "sin":
        lr, weight_decay = 0.001, 1e-5
    elif args.data == "sin_long":
        lr, weight_decay = 0.001, 1e-5
    elif args.data == "reddit":
        lr, weight_decay = 0.001, 0.0001
    elif args.data == "uber":
        if args.model == "thp_mix":
            lr, weight_decay = 0.001, 0.001
        elif args.model == "if":
            lr, weight_decay = 0.0001, 0.01
        else:
            raise NotImplementedError(f'Model {args.model} not implemented')
    elif args.data == "wiki":
        lr, weight_decay = 0.0001, 0.001
    elif args.data == "so_fold1":
        lr, weight_decay = 0.0001, 0.001
    elif args.data == "taxi":
        lr, weight_decay = 0.001, 0.001
    elif args.data == "mooc":
        lr, weight_decay = 0.001, 1e-5
    else:
        raise NotImplementedError(f'Data {args.data} not implemented')

    if args.train_tag:
        tpp_dir_path = os.path.join(
            log_dir, f"{args.data}_{args.model}_base_{args.train_tag}",
            f"seed{args.seed}", f"d{args.dim}_lr{lr}_wd{weight_decay}")
    else:
        tpp_dir_path = os.path.join(
            log_dir, f"{args.data}_{args.model}_base",
            f"seed{args.seed}", f"d{args.dim}_lr{lr}_wd{weight_decay}")

    n_layers=[4, 8, 16]
    n_heads=[32, 64, 128]
    multipliers=[2, 4, 8, 12]
    diff_lrs=[0.002, 0.0002, 0.00002]
    diff_weight_decays=[0.0] #, 0.00001] # (0.00001,0.000001,0.0)

    combinations = list(itertools.product(
        n_layers, n_heads, multipliers, diff_lrs, diff_weight_decays))

    best_eval_metric = np.inf
    best_eval_metric_idx = -1
    best_combination = None
    best_eval_metrics = None
    for combination in combinations:
        n_layer, n_head, multiplier, diff_lr, diff_weight_decay = combination
        diffuser_dir_path = os.path.join(
            tpp_dir_path,
            f"{args.diffuser}_l{n_layer}_h{n_head}_m{multiplier}_lr{diff_lr}_wd{format_float(diff_weight_decay)}_{args.eval_tag}")
        csv_path = os.path.join(
            diffuser_dir_path, "csv/version_0/metrics.csv")

        if not os.path.exists(csv_path):
            print(f'csv_path: {csv_path} does not exist')
            continue

        df = pd.read_csv(csv_path)
        try:
            eval_metrics = np.array(df[eval_metric])
        except:
            print(csv_path)
            print(f'df for {combination} does not have {eval_metric} yet. Skipping')
            continue
        orig_len = len(eval_metrics)

        eval_metrics = eval_metrics[~np.isnan(eval_metrics)]
        new_len = len(eval_metrics)

        if new_len <= 0:
            print('eval_metrics is empty. Skipping')
            print(csv_path)
            continue

        #if orig_len > 2 * new_len + 10:
        #    print(f'original: {orig_len}, new: {new_len} -> probably wrong')
        #print(eval_metrics[:100], eval_metrics.shape, combination)

        min_eval_metric_idx = np.argmin(eval_metrics)
        min_eval_metric = eval_metrics[min_eval_metric_idx]

        if min_eval_metric < best_eval_metric:
            best_eval_metric = min_eval_metric
            best_eval_metric_idx = min_eval_metric_idx
            best_combination = combination
            best_eval_metrics = eval_metrics

        print(f'{combination}: {len(eval_metrics)} epoch -> {min_eval_metric}')

    print(f'Best eval metric: {best_eval_metric} around epoch {best_eval_metric_idx} with combination: {best_combination}')







