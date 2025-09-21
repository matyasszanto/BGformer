import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import pathlib

from utils import dataloader, connect_gt_and_pred, calculate_error_distributions

from statsforecast import StatsForecast
from neuralforecast import NeuralForecast
from neuralforecast.losses.numpy import mae, mse, rmse


def run(config, models = None, debug = False):
    
    if debug:
        # load pretrained model
        models = NeuralForecast.load(path='models/2025_09_21_08_23_TimesNet_3_3_400000')
    
    elif models == None:
        print('No models were supplied, exiting!')
        plot = plt.imshow(np.zeros((200, 200)))
        return plot

    horizon = config['horizon']
    snippet_length = config['snippet_length']
    input_window_length = snippet_length - horizon

    data = dataloader(config=config, train=False, valid=False)
    Y_df = data.find_all_continuous_snippets(hours=snippet_length)

    # drop last horizon length y values for prediction
    ids_to_drop = [i for i in range(len(Y_df)) if i % snippet_length >= snippet_length-horizon]
    ids_to_drop_for_comparison = [i for i in range(len(Y_df)) if i % snippet_length < snippet_length-horizon]

    Y_df_gt_window = Y_df.drop(ids_to_drop)
    Y_df_gt_horizon = Y_df.drop(ids_to_drop_for_comparison)

    chunk_size = 32 * input_window_length  # Process 32 complete windows at a time
    if len(Y_df_gt_window) % chunk_size != 0:
        final_length = (len(Y_df_gt_window) // chunk_size) * chunk_size
        Y_df_gt_window_chunked = Y_df_gt_window.iloc[:final_length].reset_index(drop=True)
    else:
        Y_df_gt_window_chunked = Y_df_gt_window

    # predict
    Y_hat_df_2 = models.predict(df=Y_df_gt_window_chunked).reset_index()

    # rename median column names to the model name and introduce lower bound (0)
    for model in models.models:
        Y_hat_df_2.rename(columns={f'{model}-median': str(model)}, inplace=True)
        Y_hat_df_2[str(model)] = Y_hat_df_2[str(model)].clip(lower=0.0)

    if config['normalize']:

        # denormalize GT values
        Y_df_gt_horizon['y'] = data.scaler.inverse_transform(Y_df_gt_horizon['y'].values.reshape(-1,1))
        Y_df['y'] = data.scaler.inverse_transform(Y_df['y'].values.reshape(-1,1))
        
        # denormalize predicted values
        for model in models.models:
            Y_hat_df_2[str(model)] = data.scaler.inverse_transform(Y_hat_df_2[str(model)].values.reshape(-1,1))
            Y_hat_df_2[f'{model}-lo-0.95'] = data.scaler.inverse_transform(Y_hat_df_2[f'{model}-lo-0.95'].values.reshape(-1,1))
            Y_hat_df_2[f'{model}-hi-0.95'] = data.scaler.inverse_transform(Y_hat_df_2[f'{model}-hi-0.95'].values.reshape(-1,1))

    # calculate metrics
    metrics_cols = ['Model Name', 'MAE', 'MSE', 'RMSE']
    metrics = pd.DataFrame(columns=metrics_cols)
    for model in models.models:
        model_metrics = {metrics_cols[0]: str(model)}
        model_metrics['MAE'] = mae(Y_hat_df_2[str(model)], Y_df_gt_horizon['y'])
        model_metrics['MSE'] = mse(Y_hat_df_2[str(model)], Y_df_gt_horizon['y'])
        model_metrics['RMSE'] = rmse(Y_hat_df_2[str(model)], Y_df_gt_horizon['y'])

        model_metrics_df = pd.DataFrame(model_metrics, index=[0])
        metrics = pd.concat([metrics, model_metrics_df], ignore_index=True)


    # plot
    df_to_plot_2 = connect_gt_and_pred(df_gt=Y_df, df_pred=Y_hat_df_2, snippet_length=snippet_length, horizon=horizon)
    prediction_graphs = StatsForecast.plot(Y_df, df_to_plot_2.drop(columns=['y', 'cutoff', 'level_0']), max_insample_length=1260)
    
    if not debug:
        histograms, means_stds_df = calculate_error_distributions(Y_df_gt_horizon, Y_hat_df_2, models=models, horizon=horizon)

    if debug:
        prediction_graphs.savefig(fname=f'test_{models.models[0]}_out.png')


    print()

    print('\n\n\n')
    print('Errors:')
    print(metrics.head())

    print('\n\n\n')
    print("----------------------------")
    print("--------Testing done--------")
    print("----------------------------")
    print('\n\n\n')

    
    if not debug:
        return prediction_graphs, metrics, histograms, means_stds_df
    else:
        return 0



if __name__ == "__main__":

    debug_config = {
        "models": "NHITS",
        'train_dataset': 'ICU_train',
        'test_dataset': 'ICU_test_10',
        'normalize': True,
        'horizon': 3,
        'snippet_length': 6,
        'valid': False,
    }
    run(config=debug_config, debug=True)