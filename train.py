import pandas as pd

from utils import to_datetime, connect_gt_and_pred

from statsforecast import StatsForecast
from neuralforecast import NeuralForecast
from neuralforecast import core

def run(config=None, debug=False):

    if debug:
        config = {"models": "NHITS"}

    elif config['models'] == []:
        print("Error: no model configured!")
        return None, None

    elif 'TN_topk' not in config.keys():
        config['TN_topk'] = 3
    
    Y_df = pd.read_csv("preliminary/full_dataset_hourly.csv")
    timestamps = Y_df['ds']
    new_timestamps = timestamps.apply(to_datetime)
    Y_df['ds'] = new_timestamps

    uids = Y_df['unique_id'].unique()
    Y_df = Y_df.query('unique_id in @uids').reset_index(drop=True)
    
    freq = 'h' # h or min
    horizon = 3 if freq=='h' else 90
    model_strings_array = [config['models']] if isinstance(config['models'], str) else config['models']
    models_array = []
    for model in model_strings_array:
        model_object = getattr(core, model)
        if model == "iTransformer":
            models_array.append(model_object(h=horizon, input_size=20, max_steps=1, n_series=1))
        elif model == "TimesNet":
            models_array.append(model_object(h=horizon, input_size=20, max_steps=1, top_k=config['TN_topk']))
        else:
            models_array.append(model_object(h=horizon, input_size=20, max_steps=1))
    
    nf = NeuralForecast(
        models=models_array,
        freq=freq
    )

    df_cv = nf.cross_validation(Y_df, n_windows=1)

    df_to_plot = connect_gt_and_pred(df_gt=Y_df, df_pred=df_cv)

    plot = StatsForecast.plot(Y_df, df_to_plot.drop(columns=['y', 'cutoff']), max_insample_length=1260)

    print('\n\n\n')
    print("-----------------------------")
    print("--------Training done--------")
    print("-----------------------------")
    print('\n\n\n')

    return nf, plot

if __name__ == '__main__':
    run(debug=True)
    