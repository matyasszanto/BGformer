import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import pathlib

from utils import to_datetime, connect_gt_and_pred

from statsforecast import StatsForecast
from neuralforecast import NeuralForecast
from neuralforecast.losses.numpy import mae, mse, rmse


def run(models = None, debug = False):
    
    if debug:
        # load pretrained model
        models = NeuralForecast.load(path='models/ohio_train_0')
    
    elif models == None:
        print('No models were supplied, exiting!')
        plot = plt.imshow(np.zeros((200, 200)))
        return plot
    

    # read file
    basepath = pathlib.Path(__file__).parent.resolve()
    csv_path = pathlib.Path.joinpath(basepath, 'data/ICU_data/hourly_ICU_for_nf.csv')
    Y_df = pd.read_csv(csv_path)
    Y_df.drop(columns=['Unnamed: 0'], inplace=True)


    # set up timestamps
    timestamps = Y_df['ds']
    new_timestamps = timestamps.apply(to_datetime)
    Y_df['ds'] = new_timestamps
    Y_df.rename(columns={'bg': 'y'}, inplace=True)


    # drop last 3 y values for prediction
    ids_to_drop = []
    ids_to_drop_for_comparison = []
    for i, row in enumerate(Y_df.iterrows()):
        if row[1]['ds'].hour>=21:
            ids_to_drop.append(i)
        if row[1]['ds'].hour<21:
            ids_to_drop_for_comparison.append(i)
    Y_df_window = Y_df.drop(ids_to_drop)
    Y_df_gt = Y_df.drop(ids_to_drop_for_comparison)

    # predict
    Y_hat_df_2 = models.predict(df=Y_df_window).reset_index()
    

    # calculate metrics
    metrics_cols = ['Model Name', 'MAE', 'MSE', 'RMSE']
    metrics = pd.DataFrame(columns=metrics_cols)
    for model in models.models:
        model_metrics = {metrics_cols[0]: str(model)}
        model_metrics['MAE'] = mae(Y_hat_df_2[str(model)], Y_df_gt['y'])
        model_metrics['MSE'] = mse(Y_hat_df_2[str(model)], Y_df_gt['y'])
        model_metrics['RMSE'] = rmse(Y_hat_df_2[str(model)], Y_df_gt['y'])

        model_metrics_df = pd.DataFrame(model_metrics, index=[0])
        metrics = pd.concat([metrics, model_metrics_df], ignore_index=True)


    # plot
    if not debug:
        df_to_plot_2 = connect_gt_and_pred(df_gt=Y_df, df_pred=Y_hat_df_2)
        plot = StatsForecast.plot(Y_df, df_to_plot_2.drop(columns=['y', 'cutoff']), max_insample_length=1260)

    print('\n\n\n')
    print('Errors:')
    print(metrics.head())

    print('\n\n\n')
    print("----------------------------")
    print("--------Testing done--------")
    print("----------------------------")
    print('\n\n\n')

    
    if not debug:
        return plot, metrics
    else:
        return 0


if __name__ == "__main__":
    run(debug=True)