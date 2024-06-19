import os
import pandas as pd
import numpy as np
import argparse
from functools import partial



def loss_df(metric, loss, task_name, lrs, weight_decays, flood_levels, model_dims, gammas, distill_weights, aux_num):
    log_dir = os.path.join(f"{os.getcwd()}", "results")
    if metric == 'acc':
        result_df = pd.DataFrame(columns=['lr', 'wd', 'fl', 'mdim', 'gamma', 'dw', 'acc', 'loss', 'test_acc', 'test_loss'])
    else:
        raise NotImplementedError(f'Metric {metric} not implemented')

    # cls
    if loss == 'cls':
        for lr in lrs:
            for weight_decay in weight_decays:
                output_dir = os.path.join(
                    f"{log_dir}", f"{task_name}", f"seed{args.seed}", f"lr{lr}_wd{weight_decay}")
                csv_path = os.path.join(
                        output_dir, 'csv', 'version_0', 'metrics.csv')
                if not os.path.exists(csv_path):
                    continue

                print(csv_path)
                try:
                    df = pd.read_csv(csv_path)
                    indices = np.where(np.logical_not(np.isnan(df[f'{args.eval_type}/{metric}_best'])))[0]
                except:
                    continue

                if len(indices) <= 0:
                    acc = np.inf
                elif args.eval_type == 'test':
                    idx_best = indices[-1]
                    df_best = df.iloc[idx_best]
                    acc = df_best[f'{args.eval_type}/acc_best']
                    loss_with_acc_best = df_best[f'{args.eval_type}/loss_with_acc_best']
                else:
                    idx_best = indices[-1]
                    df_best = df.iloc[idx_best]
                    if metric == 'acc':
                        acc = df_best[f'{args.eval_type}/acc_best']
                        loss_with_acc_best = df_best[f'{args.eval_type}/loss_with_acc_best']
                    idx_val_best = np.where(df[f'{args.eval_type}/{metric}'] == acc)[0][0]
                    #print(f'Best val epoch for {loss}: {int(df.iloc[idx_val_best].epoch)}')

                try:
                    test_indices = np.where(np.logical_not(np.isnan(df[f'test/{metric}'])))[0]
                    test_idx = test_indices[-1]
                    test_df = df.iloc[test_idx]
                    test_acc = test_df[f'test/acc']
                    test_loss = test_df[f'test/loss']
                except:
                    test_acc, test_loss = np.inf, np.inf

                result_df = pd.concat([result_df, pd.DataFrame([{
                    'lr': lr, 'wd': weight_decay, 'fl': None, 'mdim': None, 'gamma': None, 'dw': None,
                    'acc': np.round(acc, 4),
                    'loss': np.round(loss_with_acc_best, 4),
                    'test_acc': np.round(test_acc, 4),
                    'test_loss': np.round(test_loss, 4)
                }])], ignore_index=True)

    # flood or iflood
    elif loss in ['flood', 'iflood']:
        for lr in lrs:
            for weight_decay in weight_decays:
                for flood_level in flood_levels:
                    output_dir = os.path.join(
                        f"{log_dir}", f"{task_name}", f"seed{args.seed}", f"lr{lr}_wd{weight_decay}_fl{flood_level}")
                    csv_path = os.path.join(
                        output_dir, 'csv', 'version_0', 'metrics.csv')
                    if not os.path.exists(csv_path):
                        continue

                    try:
                        df = pd.read_csv(csv_path)
                        indices = np.where(np.logical_not(np.isnan(df[f'{args.eval_type}/{metric}_best'])))[0]
                    except:
                        continue

                    if len(indices) <= 0:
                        acc = np.inf
                    elif args.eval_type == 'test':
                        idx_best = indices[-1]
                        df_best = df.iloc[idx_best]
                        acc = df_best[f'{args.eval_type}/acc_best']
                        loss_with_acc_best = df_best[f'{args.eval_type}/loss_with_acc_best']
                    else:
                        idx_best = indices[-1]
                        df_best = df.iloc[idx_best]
                        if metric == 'acc':
                            acc = df_best[f'{args.eval_type}/acc_best']
                            loss_with_acc_best = df_best[f'{args.eval_type}/loss_with_acc_best']
                        idx_val_best = np.where(df[f'{args.eval_type}/{metric}'] == acc)[0][0]
                        #print(f'Best val epoch for {loss} with {flood_level}: {int(df.iloc[idx_val_best].epoch)}')

                    try:
                        test_indices = np.where(
                            np.logical_not(np.isnan(df[f'test/{metric}'])))[0]
                        test_idx = test_indices[-1]
                        test_df = df.iloc[test_idx]
                        test_acc = test_df[f'test/acc']
                        test_loss = test_df[f'test/loss']
                    except:
                        test_acc, test_loss = np.inf, np.inf

                    result_df = pd.concat([result_df, pd.DataFrame([{
                        'lr': lr, 'wd': weight_decay, 'fl': flood_level, 'mdim': None, 'gamma': None, 'dw': None,
                        'acc': np.round(acc, 4),
                        'loss': np.round(loss_with_acc_best, 4),
                        'test_acc': np.round(test_acc, 4),
                        'test_loss': np.round(test_loss, 4)
                    }])], ignore_index=True)

    # adaflood
    elif loss == 'adaflood':
        for lr in lrs:
            for weight_decay in weight_decays:
                for model_dim in model_dims:
                    for gamma in gammas:
                        output_dir = os.path.join(
                            f"{log_dir}", f"{task_name}", f"seed{args.seed}", f"lr{lr}_wd{weight_decay}_mdim{model_dim}_gamma{gamma}_aux{aux_num}")
                        csv_path = os.path.join(
                            output_dir, 'csv', 'version_0', 'metrics.csv')
                        if not os.path.exists(csv_path):
                            #import IPython; IPython.embed()
                            continue

                        try:
                            df = pd.read_csv(csv_path)
                            indices = np.where(np.logical_not(np.isnan(df[f'{args.eval_type}/{metric}_best'])))[0]
                        except:
                            continue

                        if len(indices) <= 0:
                            acc = np.inf
                        elif args.eval_type == 'test':
                            idx_best = indices[-1]
                            df_best = df.iloc[idx_best]
                            acc = df_best[f'{args.eval_type}/acc_best']
                            loss_with_acc_best = df_best[f'{args.eval_type}/loss_with_acc_best']
                        else:
                            idx_best = indices[-1]
                            df_best = df.iloc[idx_best]
                            if metric == 'acc':
                                acc = df_best[f'{args.eval_type}/acc_best']
                                loss_with_acc_best = df_best[f'{args.eval_type}/loss_with_acc_best']
                            idx_val_best = np.where(df[f'{args.eval_type}/{metric}'] == acc)[0][0]
                            #print(f'Best val epoch for {loss} with {gamma}: {int(df.iloc[idx_val_best].epoch)}')

                        try:
                            test_indices = np.where(
                                np.logical_not(np.isnan(df[f'test/{metric}'])))[0]
                            test_idx = test_indices[-1]
                            test_df = df.iloc[test_idx]
                            test_acc = test_df[f'test/acc']
                            test_loss = test_df[f'test/loss']
                        except:
                            test_acc, test_loss = np.inf, np.inf

                        result_df = pd.concat([result_df, pd.DataFrame([{
                            'lr': lr, 'wd': weight_decay, 'fl': None, 'mdim': model_dim, 'gamma': gamma, 'dw': None,
                            'acc': np.round(acc, 4),
                            'loss': np.round(loss_with_acc_best, 4),
                            'test_acc': np.round(test_acc, 4),
                            'test_loss': np.round(test_loss, 4)
                        }])], ignore_index=True)

    # knowledge distillation
    elif loss == 'kd':
        for lr in lrs:
            for weight_decay in weight_decays:
                for model_dim in model_dims:
                    for gamma in gammas:
                        for distill_weight in distill_weights:
                            output_dir = os.path.join(
                                f"{log_dir}", f"{task_name}", f"seed{args.seed}", f"lr{lr}_wd{weight_decay}_mdim{model_dim}_gamma{gamma}_dw{distill_weight}_aux{aux_num}")
                            csv_path = os.path.join(
                                output_dir, 'csv', 'version_0', 'metrics.csv')
                            if not os.path.exists(csv_path):
                                #print(csv_path)
                                #import IPython; IPython.embed()
                                continue

                            try:
                                df = pd.read_csv(csv_path)
                                indices = np.where(np.logical_not(np.isnan(df[f'{args.eval_type}/{metric}_best'])))[0]
                            except:
                                continue

                            if len(indices) <= 0:
                                acc = np.inf
                            elif args.eval_type == 'test':
                                idx_best = indices[-1]
                                df_best = df.iloc[idx_best]
                                acc = df_best[f'{args.eval_type}/acc_best']
                                loss_with_acc_best = df_best[f'{args.eval_type}/loss_with_acc_best']
                            else:
                                idx_best = indices[-1]
                                df_best = df.iloc[idx_best]
                                if metric == 'acc':
                                    acc = df_best[f'{args.eval_type}/acc_best']
                                    loss_with_acc_best = df_best[f'{args.eval_type}/loss_with_acc_best']
                                idx_val_best = np.where(df[f'{args.eval_type}/{metric}'] == acc)[0][0]
                                #print(f'Best val epoch for {loss} with {gamma}: {int(df.iloc[idx_val_best].epoch)}')

                            try:
                                test_indices = np.where(
                                    np.logical_not(np.isnan(df[f'test/{metric}'])))[0]
                                test_idx = test_indices[-1]
                                test_df = df.iloc[test_idx]
                                test_acc = test_df[f'test/acc']
                                test_loss = test_df[f'test/loss']
                            except:
                                test_acc, test_loss = np.inf, np.inf

                            result_df = pd.concat([result_df, pd.DataFrame([{
                                'lr': lr, 'wd': weight_decay, 'fl': None, 'mdim': model_dim, 'gamma': gamma, 'dw': distill_weight,
                                'acc': np.round(acc, 4),
                                'loss': np.round(loss_with_acc_best, 4),
                                'test_acc': np.round(test_acc, 4),
                                'test_loss': np.round(test_loss, 4)
                            }])], ignore_index=True)


    else:
        raise NotImplementedError(f"Loss {args.loss} is not implemented")

    sorted_df = result_df.sort_values(f'{metric}', ascending=False)
    return sorted_df

def summarize_dfs(loss_dfs, loss_names):
    columns = ['method'] + loss_dfs[0].columns.tolist()
    summary_df = pd.DataFrame(columns=columns)

    #import IPython; IPython.embed()
    for df, name in zip(loss_dfs, loss_names):
        if df.empty:
            values = [name] + [np.nan] * 4
        else:
            values = [name] + [val for val in df.iloc[0]]
        new_df = pd.DataFrame([{col_name: val for col_name, val in zip(columns, values)}])
        try:
            summary_df = pd.concat([summary_df, new_df], ignore_index=True)
        except:
            import IPython; IPython.embed()

    return summary_df



if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", required=True,
                        choices=["svhn", "cifar10", "cifar100", "imagenet100", "cars"])
    parser.add_argument("-m", "--model", required=True,
                        choices=["resnet18", "resnet34"])
    parser.add_argument("-l", "--loss",
                        choices=["tpp", "flood", "iflood", "adaflood", "kd"])
    parser.add_argument("-a", "--alpha", type=float, default=0.0)
    parser.add_argument("-i", "--imb", type=float, default=1.0)
    parser.add_argument("-t", "--tag", default="test")
    parser.add_argument("-s", "--seed", default=1)
    parser.add_argument("-e", "--eval_type", default="val",
                        choices=["val", "test"])
    parser.add_argument("-n", "--aux_num", required=True)
    parser.add_argument("-v", "--save", type=bool, default=False)
    args = parser.parse_args()

    if args.loss:
        losses = [args.loss]
    else:
        losses = ["cls", "flood", "iflood", "adaflood", "kd"]

    metrics = ["acc"]

    if args.data == 'cifar10':
        if args.alpha == 0.0:
            lrs, weight_decays = [0.1], [0.0] #[0.0001] # [0.0]
        else:
            lrs, weight_decays = [0.1], [0.0]
        #flood_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
        #                0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]
        #flood_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.60, 0.70, 0.80]
    elif args.data == 'cifar100':
        if args.alpha == 0.0:
            lrs, weight_decays = [0.1], [0.0] #[0.0001] #[0.0] # [0.0001]
        else:
            lrs, weight_decays = [0.1], [0.0]
        #flood_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
        #                0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]
    elif args.data == 'svhn':
        if args.alpha == 0.0:
            lrs, weight_decays = [0.1], [0.0] # [0.0001]
        else:
            lrs, weight_decays = [0.1], [0.0]
        #flood_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
        #                0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]
    elif args.data == 'imagenet100':
        lrs, weight_decays = [0.1], [0.0001]
        #flood_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    elif args.data == 'cars':
        lrs, weight_decays = [0.1], [0.0]
        #flood_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    else:
        lrs = [0.1, 0.01, 0.001]
        weight_decays = [0.01, 0.001, 0.0001, 1e-5]
        #flood_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    #flood_levels = [0.0,0.0001,0.0003,0.001,0.003,0.01,0.03,0.1,0.3,1.0]
    model_dims = [64]
    #gammas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    flood_levels = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,
                    0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                    0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]
    gammas = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,
              0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.90, 0.95, 1.0]
    distill_weights = [1.0, 3.0, 5.0, 7.0, 9.0]

    print(f'Data: {args.data}')
    summary_dfs = []
    for metric in metrics:
        loss_dfs = []
        for loss in losses:
            task_name = f"{args.data}_{args.model}_alpha{args.alpha}_imb{args.imb}_{loss}_{args.tag}"

            df = loss_df(metric, loss, task_name, lrs, weight_decays,
                         flood_levels, model_dims, gammas, distill_weights, args.aux_num)
            loss_dfs.append(df)

            print(f'Loss: {loss}, Metric: {metric}')
            print(df.to_string())
            print('======================================================')

        if len(loss_dfs) > 1:
            summary_df = summarize_dfs(loss_dfs, losses)
            summary_dfs.append(summary_df)

    print(f'Data: {args.data}')
    for metric, summary_df in zip(metrics, summary_dfs):
        print(f'Metric: {metric}')
        print(summary_df.to_string())
        print('======================================================')

        if args.save:
            file_path = os.path.join(
                'summary', f"{args.data}_{args.model}_alpha{args.alpha}_imb{args.imb}_{args.tag}_aux{args.aux_num}_{metric}")
            summary_df.to_csv(file_path)

