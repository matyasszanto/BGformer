import pandas as pd
import pathlib
import os
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np

from utils import dataloader

from statsforecast import StatsForecast
from neuralforecast import NeuralForecast
from neuralforecast import core

def run(config=None, debug=False):

    if debug:
        config = {
            "models": "TimesNet",
            'train_dataset': 'ICU_test_10',
            'test_dataset': 'ICU_test_10',
            'normalize': True,
            'max_steps': 3000,
            'val_check_steps': 1500,
            'horizon': 3,
            'snippet_length': 6,
            'TN_topk': 2,
            'enable_checkpointing': True,
            'early_stop_patience_steps': 5,
            'padding': True,
        }

    if config['models'] == []:
        print("Error: no model configured!")
        return None, None

    if 'TN_topk' not in config.keys():
        config['TN_topk'] = 3

    if 'enable_checkpointing' not in config.keys():
        config['enable_checkpointing'] = True

    if 'padding' not in config.keys():
        config['padding'] = False
        
    model_strings_array = [config['models']] if isinstance(config['models'], str) else config['models']
    models_array = []
    max_steps = config['max_steps']
    val_check_steps = config['val_check_steps']
    padding = config['padding']
    early_stop_patience_steps = config['early_stop_patience_steps'] if padding else -1
    
    # Configure trainer to save only the best model based on validation loss
    from pytorch_lightning.callbacks import ModelCheckpoint
    
    # Create checkpoint callback that will be configured with the correct directory later
    checkpoint_callback = None
    trainer_kwargs = {
        'enable_checkpointing': True
    }

    data = dataloader(config=config, train=True)
    Y_df = data.find_all_continuous_snippets(hours=config['snippet_length'])
    
    horizon_hours = config['horizon'] # length of horizon in hours
    
    freq = 'h' # h or min
    horizon = horizon_hours if freq=='h' else 90
    
    
    for model in model_strings_array:
        model_object = getattr(core, model)
        if model == "iTransformer":
            models_array.append(model_object(h=horizon, 
                                             input_size=config['snippet_length']-horizon,
                                             max_steps=max_steps,
                                             n_series=1, 
                                             val_check_steps=val_check_steps, 
                                             start_padding_enabled=padding,
                                             early_stop_patience_steps=early_stop_patience_steps, 
                                             **trainer_kwargs
                                             )
                                )
        elif model == "TimesNet":
            models_array.append(model_object(h=horizon, 
                                             input_size=config['snippet_length']-horizon,
                                             max_steps=max_steps, 
                                             top_k=config['TN_topk'], 
                                             val_check_steps=val_check_steps, 
                                             start_padding_enabled=padding,
                                             early_stop_patience_steps=early_stop_patience_steps,
                                             **trainer_kwargs
                                             )
                                )
        else:
            models_array.append(model_object(h=horizon, 
                                             input_size=config['snippet_length']-horizon,
                                             max_steps=max_steps, 
                                             val_check_steps=val_check_steps, 
                                             start_padding_enabled=padding,
                                             early_stop_patience_steps=early_stop_patience_steps,
                                             **trainer_kwargs
                                             )
                                )
    
    nf = NeuralForecast(
        models=models_array,
        freq=freq
    )
    
    val_size = config['horizon'] if padding else 0
    
    # create directory for saving model
    train_id_string = dt.datetime.strftime(dt.datetime.now(), '%Y_%m_%d_%H_%M')+'_'+config['models']+'_'+str(config['snippet_length']-config['horizon'])+'_'+str(config['horizon'])+'_'+str(config['max_steps'])
    if config['enable_checkpointing']:
        os.makedirs(f'models/{train_id_string}')
        checkpoint_dir = f'models/{train_id_string}'
        
        # Now configure the checkpoint callback with the correct directory
        # Try different metric names that NeuralForecast might use
        if val_size > 0:
            # Try validation loss with different possible names
            monitor_metric = 'val_loss'
            filename_template = 'best_model_{epoch:02d}_{val_loss:.4f}'
        else:
            # Try training loss with different possible names
            monitor_metric = 'train_loss'
            filename_template = 'best_model_{epoch:02d}_{train_loss:.4f}'
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,  # Save directly to our desired location
            monitor=monitor_metric,  # Monitor validation or training loss
            mode='min',             # Save when loss decreases
            save_top_k=1,           # Save only the best model
            save_last=False,        # Don't save the last checkpoint
            filename=filename_template,
            verbose=True
        )
        
        trainer_kwargs['callbacks'] = [checkpoint_callback]
    
    # _ = nf.cross_validation(Y_df, n_windows=1)
    
    
    # Debug: Print checkpoint configuration
    if config['enable_checkpointing'] and checkpoint_callback and debug:
        print(f"Checkpoint callback configured:")
        print(f"  - Directory: {checkpoint_callback.dirpath}")
        print(f"  - Monitor: {checkpoint_callback.monitor}")
        print(f"  - Mode: {checkpoint_callback.mode}")
        print(f"  - Save top k: {checkpoint_callback.save_top_k}")
        print(f"  - Validation size: {val_size}")
        print(f"  - Padding enabled: {padding}")
        print(f"  - Filename template: {checkpoint_callback.filename}")
    
    nf.fit(Y_df, val_size = val_size)

    # Debug: Check what metrics are available in the model
    if config['enable_checkpointing'] and checkpoint_callback and debug:
        print(f"\nDebug: Checking available metrics in model...")
        try:
            # Check if the model has a trainer and what callbacks are attached
            if hasattr(nf.models[0], 'trainer') and nf.models[0].trainer:
                print(f"  - Trainer found: {type(nf.models[0].trainer)}")
                print(f"  - Callbacks: {[type(cb).__name__ for cb in nf.models[0].trainer.callbacks]}")
                
                # Check if ModelCheckpoint is in the callbacks
                model_checkpoint_callbacks = [cb for cb in nf.models[0].trainer.callbacks if 'ModelCheckpoint' in str(type(cb))]
                if model_checkpoint_callbacks:
                    print(f"  - ModelCheckpoint callbacks found: {len(model_checkpoint_callbacks)}")
                    for cb in model_checkpoint_callbacks:
                        print(f"    - Monitor: {cb.monitor}")
                        print(f"    - Best path: {cb.best_model_path}")
                        print(f"    - Best score: {cb.best_model_score}")
                else:
                    print(f"  - No ModelCheckpoint callbacks found!")
                    
                # Check what metrics are available in the trainer
                if hasattr(nf.models[0].trainer, 'logged_metrics'):
                    print(f"  - Logged metrics: {list(nf.models[0].trainer.logged_metrics.keys())}")
            else:
                print(f"  - No trainer found in model")
        except Exception as e:
            print(f"  - Error checking trainer: {e}")

    # Try to extract training and validation loss from the model(s)
    train_graph = None
    best_model_info = None
    
    try:
        train_loss = nf.models[0].train_trajectories
        df_train_loss = pd.DataFrame(train_loss, columns=['x', 'y'])
        val_loss = nf.models[0].valid_trajectories
        df_val_loss = pd.DataFrame(val_loss, columns=['x', 'y'])
        
        if train_loss is not None and val_loss is not None:
            plt.figure(figsize=(8, 5))
            plt.plot(df_train_loss['x'], df_train_loss['y'], label='Training Loss')
            plt.plot(df_val_loss['x'], df_val_loss['y'], label='Validation Loss')
            plt.xlabel('Steps')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.tight_layout()
            train_graph = plt.gcf()
            plt.close()
            
            # Find the best model based on validation loss (if available) or training loss
            if val_loss is not None and len(val_loss) > 0:
                # Use validation loss to find best model
                best_val_loss_idx = np.argmin([point[1] for point in val_loss])
                best_val_loss = val_loss[best_val_loss_idx][1]
                best_step = val_loss[best_val_loss_idx][0]
                best_model_info = {
                    'metric': 'validation_loss',
                    'value': best_val_loss,
                    'step': best_step,
                    'index': best_val_loss_idx
                }
            elif train_loss is not None and len(train_loss) > 0:
                # Fallback to training loss
                best_train_loss_idx = np.argmin([point[1] for point in train_loss])
                best_train_loss = train_loss[best_train_loss_idx][1]
                best_step = train_loss[best_train_loss_idx][0]
                best_model_info = {
                    'metric': 'training_loss',
                    'value': best_train_loss,
                    'step': best_step,
                    'index': best_train_loss_idx
                }
        else:
            train_graph = None
    except Exception as e:
        train_graph = None
        print(f"Error extracting loss trajectories: {e}")

    # Save the best model based on loss analysis
    if config['enable_checkpointing'] and best_model_info:
        print(f"\nBest Model Analysis:")
        print(f"  - Best {best_model_info['metric']}: {best_model_info['value']:.6f}")
        print(f"  - Best step: {best_model_info['step']}")
        print(f"  - Best index: {best_model_info['index']}")
        
        # Save the best model to the desired location
        best_model_path = f'models/{train_id_string}/best_model_{best_model_info["step"]:02d}_{best_model_info["value"]:.4f}'
        try:
            nf.save(best_model_path, model_index=[0], save_dataset=False, overwrite=True)
            print(f"  - Best model saved to: {best_model_path}")
        except Exception as e:
            print(f"  - Error saving best model: {e}")
    elif config['enable_checkpointing']:
        print(f"\nWarning: Could not determine best model - saving final model instead")
        try:
            final_model_path = f'models/{train_id_string}/final_model'
            nf.save(final_model_path, model_index=[0], save_dataset=False, overwrite=True)
            print(f"  - Final model saved to: {final_model_path}")
        except Exception as e:
            print(f"  - Error saving final model: {e}")

    print('\n\n\n')
    print("-----------------------------")
    print("--------Training done--------")
    print("-----------------------------")
    print('\n\n\n')

    return nf, train_graph

if __name__ == '__main__':
    _,  graph = run(debug=True)
    graph.savefig(fname='train_graph.png')
    