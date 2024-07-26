import pandas as pd
import pathlib

from utils import to_datetime, find_all_continuous_snippets

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

# read data

    # ICU data
    basepath = pathlib.Path(__file__).parent.resolve()
    csv_path = pathlib.Path.joinpath(basepath, 'data/ICU_data/hourly_ICU_for_nf.csv')

    # OhioT1DM data
    # csv_path = "data/OhioT1DM/full_dataset_hourly.csv"

    Y_df = pd.read_csv(csv_path)
    if 'Unnamed: 0' in Y_df.columns:
        Y_df.drop(columns=['Unnamed: 0'], inplace=True)


    timestamps = Y_df['ds']
    new_timestamps = timestamps.apply(to_datetime)
    Y_df['ds'] = new_timestamps
    if 'bg' in Y_df.columns:
        Y_df.rename(columns={'bg': 'y'}, inplace=True)

    uids = Y_df['unique_id'].unique()
    Y_df = Y_df.query('unique_id in @uids').reset_index(drop=True)

    Y_df = find_all_continuous_snippets(Y_df)
    
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

    _ = nf.cross_validation(Y_df, n_windows=1)

    print('\n\n\n')
    print("-----------------------------")
    print("--------Training done--------")
    print("-----------------------------")
    print('\n\n\n')

    return nf

if __name__ == '__main__':
    run(debug=True)
    