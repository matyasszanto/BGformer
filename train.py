import pandas as pd
import pathlib

from utils import dataloader

from statsforecast import StatsForecast
from neuralforecast import NeuralForecast
from neuralforecast import core

def run(config=None, debug=False):

    if debug:
        config = {
            "models": "NHITS",
            'train_dataset': 'ICU',
            'test_dataset': 'Ohio',
            'normalize': True,
        }

    elif config['models'] == []:
        print("Error: no model configured!")
        return None, None

    elif 'TN_topk' not in config.keys():
        config['TN_topk'] = 3

    data = dataloader(config=config, train=True)
    Y_df = data.find_all_continuous_snippets()
    
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
    