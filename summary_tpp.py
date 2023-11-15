import os
import pandas as pd
import numpy as np
import argparse
from functools import partial



def loss_df(metric, loss, task_name, lrs, weight_decays, flood_levels, model_dims, gammas, aux_num):
    log_dir = os.path.join(f"{os.getcwd()}", "results")
    if metric == 'rmse':
        result_df = pd.DataFrame(columns=['lr', 'wd', 'fl', 'mdim', 'gamma', 'rmse', 'nll', 'acc', 'test_rmse', 'test_nll', 'test_acc'])
    elif metric == 'nll':
        result_df = pd.DataFrame(columns=['lr', 'wd', 'fl', 'mdim', 'gamma', 'nll', 'rmse', 'acc', 'test_nll', 'test_rmse', 'test_acc'])
    else:
        raise NotImplementedError(f'Metric {metric} not implemented')

    # tpp
    if loss == 'tpp':
        for lr in lrs:
            for weight_decay in weight_decays:
                output_dir = os.path.join(
                    f"{log_dir}", f"{task_name}", f"seed{args.seed}", f"lr{lr}_wd{weight_decay}")
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
                    rmse, nll, acc = np.inf, np.inf, np.inf
                elif args.eval_type == 'test':
                    idx_best = indices[-1]
                    df_best = df.iloc[idx_best]
                    rmse = df_best[f'{args.eval_type}/rmse_best']
                    nll = df_best[f'{args.eval_type}/nll_best']
                    acc = df_best[f'{args.eval_type}/acc_best']
                else:
                    idx_best = indices[-1]
                    df_best = df.iloc[idx_best]
                    if metric == 'rmse':
                        rmse = df_best[f'{args.eval_type}/rmse_best']
                        try:
                            nll = df_best[f'{args.eval_type}/nll_with_rmse_best']
                        except:
                            nll = np.inf
                    if metric == 'nll':
                        nll = df_best[f'{args.eval_type}/nll_best']
                        try:
                            rmse = df_best[f'{args.eval_type}/rmse_with_nll_best']
                        except:
                            rmse = np.inf

                    if f'{args.eval_type}/acc_with_rmse_best' in df_best:
                        acc = df_best[f'{args.eval_type}/acc_with_rmse_best']
                    else:
                        acc = np.inf

                try:
                    test_indices = np.where(
                        np.logical_not(np.isnan(df[f'test/{metric}_best'])))[0]
                    test_idx = test_indices[-1]
                    test_df = df.iloc[test_idx]
                    test_rmse = test_df[f'test/rmse_best']
                    test_nll = test_df[f'test/nll_best']
                    if f'test/acc_best' in test_df:
                        test_acc = test_df[f'test/acc_best']
                    else:
                        test_acc = np.inf
                except:
                    test_rmse, test_nll, test_acc = np.inf, np.inf, np.inf

                result_df = pd.concat([result_df, pd.DataFrame([{
                    'lr': lr, 'wd': weight_decay, 'fl': None, 'mdim': None, 'gamma': None,
                    'rmse': np.round(rmse, 4),
                    'nll': np.round(nll, 4),
                    'acc': np.round(acc, 4),
                    'test_rmse': np.round(test_rmse, 4),
                    'test_nll': np.round(test_nll, 4),
                    'test_acc': np.round(test_acc, 4)
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
                        if loss == 'iflood':
                            print(csv_path)
                        continue

                    try:
                        df = pd.read_csv(csv_path)
                        indices = np.where(np.logical_not(np.isnan(df[f'{args.eval_type}/{metric}_best'])))[0]
                    except:
                        indices = []
                        #continue

                    if len(indices) <= 0:
                        rmse, nll, acc = np.inf, np.inf, np.inf
                    elif args.eval_type == 'test':
                        idx_best = indices[-1]
                        df_best = df.iloc[idx_best]
                        rmse = df_best[f'{args.eval_type}/rmse_best']
                        nll = df_best[f'{args.eval_type}/nll_best']
                        acc = df_best[f'{args.eval_type}/acc_best']
                    else:
                        idx_best = indices[-1]
                        df_best = df.iloc[idx_best]
                        if metric == 'rmse':
                            rmse = df_best[f'{args.eval_type}/rmse_best']
                            try:
                                nll = df_best[f'{args.eval_type}/nll_with_rmse_best']
                                #acc = df_best[f'{args.eval_type}/acc_with_rmse_best']
                            except:
                                nll = np.inf
                        if metric == 'nll':
                            nll = df_best[f'{args.eval_type}/nll_best']
                            try:
                                rmse = df_best[f'{args.eval_type}/rmse_with_nll_best']
                                #acc = df_best[f'{args.eval_type}/acc_with_nll_best']
                            except:
                                rmse = np.inf

                        if f'{args.eval_type}/acc_with_rmse_best' in df_best:
                            acc = df_best[f'{args.eval_type}/acc_with_rmse_best']
                        else:
                            acc = np.inf


                    try:
                        test_indices = np.where(np.logical_not(np.isnan(df[f'test/{metric}_best'])))[0]
                        test_idx = test_indices[-1]
                        test_df = df.iloc[test_idx]
                        test_rmse = test_df[f'test/rmse_best']
                        test_nll = test_df[f'test/nll_best']
                        if f'test/acc_best' in test_df:
                            test_acc = test_df[f'test/acc_best']
                        else:
                            test_acc = np.inf
                    except:
                        test_rmse, test_nll, test_acc = np.inf, np.inf, np.inf

                    result_df = pd.concat([result_df, pd.DataFrame([{
                        'lr': lr, 'wd': weight_decay, 'fl': flood_level, 'mdim': None, 'gamma': None,
                        'rmse': np.round(rmse, 4),
                        'nll': np.round(nll, 4),
                        'acc': np.round(acc, 4),
                        'test_rmse': np.round(test_rmse, 4),
                        'test_nll': np.round(test_nll, 4),
                        'test_acc': np.round(test_acc, 4)
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
                        #csv_path = csv_path.replace('ubNone', 'ub')
                        if not os.path.exists(csv_path):
                            #print(csv_path)
                            continue

                        try:
                            df = pd.read_csv(csv_path)
                            indices = np.where(np.logical_not(np.isnan(df[f'{args.eval_type}/{metric}_best'])))[0]
                        except:
                            continue

                        if len(indices) <= 0:
                            rmse, nll, acc = np.inf, np.inf, np.inf
                        elif args.eval_type == 'test':
                            idx_best = indices[-1]
                            df_best = df.iloc[idx_best]
                            rmse = df_best[f'{args.eval_type}/rmse_best']
                            nll = df_best[f'{args.eval_type}/nll_best']
                            acc = df_best[f'{args.eval_type}/acc_best']
                        else:
                            idx_best = indices[-1]
                            df_best = df.iloc[idx_best]
                            if metric == 'rmse':
                                rmse = df_best[f'{args.eval_type}/rmse_best']
                                try:
                                    nll = df_best[f'{args.eval_type}/nll_with_rmse_best']
                                    #acc = df_best[f'{args.eval_type}/acc_with_rmse_best']
                                except:
                                    nll = np.inf
                            if metric == 'nll':
                                nll = df_best[f'{args.eval_type}/nll_best']
                                try:
                                    rmse = df_best[f'{args.eval_type}/rmse_with_nll_best']
                                    #acc = df_best[f'{args.eval_type}/acc_with_nll_best']
                                except:
                                    rmse = np.inf

                            if f'{args.eval_type}/acc_with_rmse_best' in df_best:
                                acc = df_best[f'{args.eval_type}/acc_with_rmse_best']
                            else:
                                acc = np.inf

                        try:
                            test_indices = np.where(
                                np.logical_not(np.isnan(df[f'test/{metric}_best'])))[0]
                            test_idx = test_indices[-1]
                            test_df = df.iloc[test_idx]
                            test_rmse = test_df[f'test/rmse_best']
                            test_nll = test_df[f'test/nll_best']
                            if f'test/acc_best' in test_df:
                                test_acc = test_df[f'test/acc_best']
                            else:
                                test_acc = np.inf
                        except:
                            test_rmse, test_nll, test_acc = np.inf, np.inf, np.inf

                        result_df = pd.concat([result_df, pd.DataFrame([{
                            'lr': lr, 'wd': weight_decay, 'fl': None, 'mdim': model_dim, 'gamma': gamma,
                            'rmse': np.round(rmse, 4),
                            'nll': np.round(nll, 4),
                            'acc': np.round(acc, 4),
                            'test_rmse': np.round(test_rmse, 4),
                            'test_nll': np.round(test_nll, 4),
                            'test_acc': np.round(test_acc, 4)
                        }])], ignore_index=True)
    else:
        raise NotImplementedError(f"Loss {args.loss} is not implemented")

    sorted_df = result_df.sort_values(f'{metric}', ascending=True)
    return sorted_df

def summarize_dfs(loss_dfs, loss_names):
    columns = ['method'] + loss_dfs[0].columns.tolist()
    summary_df = pd.DataFrame(columns=columns)

    for df, name in zip(loss_dfs, loss_names):
        if df.empty:
            values = [name] + [np.nan] * 7
        else:
            values = [name] + [val for val in df.iloc[0]]
        new_df = pd.DataFrame([{col_name: val for col_name, val in zip(columns, values)}])
        summary_df = pd.concat([summary_df, new_df], ignore_index=True)

    return summary_df



if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", required=True,
                        choices=["so_fold1", "mooc", "reddit", "wiki", "sin",
                                 "uber_drop", "taxi_times_jan_feb"])
    parser.add_argument("-m", "--model", required=True,
                        choices=["thp_mix", "if"])
    parser.add_argument("-l", "--loss",
                        choices=["tpp", "flood", "iflood", "adaflood"])
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
        losses = ["tpp", "flood", "iflood", "adaflood"]

    metrics = ["rmse", "nll"]

    if args.data == 'reddit':
        lrs, weight_decays = [0.001], [0.0001]
    elif args.data == 'uber_drop':
        if args.model == 'thp_mix':
            lrs, weight_decays = [0.001], [0.001]
        elif args.model == 'if':
            lrs, weight_decays = [0.0001], [0.01]
    elif args.data == 'wiki':
        lrs, weight_decays = [0.0001], [0.001]
    elif args.data == 'taxi_times_jan_feb':
        lrs, weight_decays = [0.001], [0.001]
    elif args.data == 'so_fold1':
        lrs, weight_decays = [0.0001], [0.001]
    else:
        lrs = [0.01, 0.001, 0.0001]
        weight_decays = [0.01, 0.001, 0.0001, 1e-5]
    #flood_levels = [-100.0,-30.0,-10.0,-3.0,-2.0,-1.0,0.0,1.0,10.0,30.0]
    #flood_levels = [-10.0,-3.0,-2.0,-1.0,0.0,1.0]
    flood_levels = [-50.0, -45.0, -40.0, -35.0, -30.0, -25.0, -20.0, -15.0, -10.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 10.0, 30.0]
    model_dims = [64]
    #upper_bounds = ['', None] + flood_levels
    gammas = [0.0, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.90, 0.95, 0.999]

    print(f'Data: {args.data}')
    summary_dfs = []
    for metric in metrics:
        loss_dfs = []
        for loss in losses:
            task_name = "{data}_{model}_{loss}_{tag}".format(
                    data=args.data, model=args.model, loss=loss, tag=args.tag)

            df = loss_df(
                metric, loss, task_name, lrs, weight_decays, flood_levels, model_dims, gammas, args.aux_num)
            loss_dfs.append(df)

            missing_indices = np.where(df['rmse'] == np.inf)[0]
            missing_df = df.iloc[missing_indices]
            missing_lrs, missing_wds = missing_df['lr'], missing_df['wd']
            missing_pairs = sorted(list(set(
                [(lr, wd) for lr, wd in zip(missing_lrs, missing_wds)])), reverse=True)

            #if loss == 'flood':
            #    import IPython; IPython.embed()

            all_indices = np.arange(len(df))
            valid_indices = np.setdiff1d(all_indices, missing_indices)
            valid_df = df.iloc[valid_indices]

            if metric == 'rmse':
                print(f'Loss: {loss}, Metric: {metric}')
                print(valid_df.to_string())
                print('======================================================')
                print(f'missing pairs: {missing_pairs}')
                print('======================================================')

        if len(loss_dfs) > 1:
            summary_df = summarize_dfs(loss_dfs, losses)
            summary_dfs.append(summary_df)

    print(f'Data: {args.data}')
    for metric, summary_df in zip(metrics, summary_dfs):
        print(f'Metric: {metric}')
        print(summary_df.to_string())
        print('======================================================')

        if args.save and metric == 'rmse':
            file_path = os.path.join(
                'summary', f"{args.data}_{args.model}_{args.tag}_aux{args.aux_num}_{metric}")
            summary_df.to_csv(file_path)

