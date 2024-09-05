import pandas as pd
import pathlib
import os
import datetime as dt

from utils import dataloader

from statsforecast import StatsForecast
from neuralforecast import NeuralForecast
from neuralforecast import core

def run(config=None, debug=False):

    if debug:
        config = {
            "models": "TimesNet",
            'train_dataset': 'ICU',
            'test_dataset': 'Ohio',
            'normalize': True,
            'max_steps': 1000,
            'val_check_steps': 1000,
            'horizon': 3,
            'snippet_length': 24,
            'TN_topk': 2,
            'enable_checkpointing': True
        }

    if config['models'] == []:
        print("Error: no model configured!")
        return None, None

    if 'TN_topk' not in config.keys():
        config['TN_topk'] = 3

    if 'enable_checkpointing' not in config.keys():
        config['enable_checkpointing'] = True

    data = dataloader(config=config, train=True)
    Y_df = data.find_all_continuous_snippets(hours=config['snippet_length'])
    
    horizon_hours = config['horizon'] # length of horizon in hours
    
    freq = 'h' # h or min
    horizon = horizon_hours if freq=='h' else 90
    
    model_strings_array = [config['models']] if isinstance(config['models'], str) else config['models']
    models_array = []
    max_steps = config['max_steps']
    val_check_steps = config['val_check_steps']
    early_stop_patience_steps = -1
    # trainer_kwargs = {'enable_checkpointing': 'True'}
    for model in model_strings_array:
        model_object = getattr(core, model)
        if model == "iTransformer":
            models_array.append(model_object(h=horizon, 
                                             input_size=config['snippet_length']-horizon,
                                             max_steps=max_steps,
                                             n_series=1, 
                                             val_check_steps=val_check_steps, 
                                             early_stop_patience_steps=early_stop_patience_steps, 
                                            #  **trainer_kwargs
                                             )
                                )
        elif model == "TimesNet":
            models_array.append(model_object(h=horizon, 
                                             input_size=config['snippet_length']-horizon,
                                             max_steps=max_steps, 
                                             top_k=config['TN_topk'], 
                                             val_check_steps=val_check_steps, 
                                             early_stop_patience_steps=early_stop_patience_steps,
                                            #  **trainer_kwargs
                                             )
                                )
        else:
            models_array.append(model_object(h=horizon, 
                                             input_size=config['snippet_length']-horizon,
                                             max_steps=max_steps, 
                                             val_check_steps=val_check_steps, 
                                             early_stop_patience_steps=early_stop_patience_steps,
                                            #  **trainer_kwargs
                                             )
                                )
    
    nf = NeuralForecast(
        models=models_array,
        freq=freq
    )

    # create directory for saving model
    train_id_string = dt.datetime.strftime(dt.datetime.now(), '%Y_%m_%d_%H_%M')
    if config['enable_checkpointing']:
        os.makedirs(f'models/{train_id_string}')

    # _ = nf.cross_validation(Y_df, n_windows=1)
    nf.fit(Y_df)

    if config['enable_checkpointing']:
        nf.save(f'models/{train_id_string}', model_index=[0], save_dataset=False, overwrite=True)

    print('\n\n\n')
    print("-----------------------------")
    print("--------Training done--------")
    print("-----------------------------")
    print('\n\n\n')

    return nf

if __name__ == '__main__':
    run(debug=True)
    