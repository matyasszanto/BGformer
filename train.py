import pandas as pd
import pathlib
import os
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import json

from utils import dataloader

from statsforecast import StatsForecast
from neuralforecast import NeuralForecast
from neuralforecast import core
from neuralforecast.losses.pytorch import MQLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import Adam, AdamW

def run(config=None, debug=False, overfit_size=None, dataframe = None):

    if debug:
        config = {
            "models": "TimesNet",
            'train_dataset': 'ICU_train',
            'test_dataset': 'ICU_test_10',
            'normalize': True,
            'max_steps': 200,
            'val_check_steps': 40,
            'horizon': 3,
            'snippet_length': 24,
            'TN_topk': 2,
            'enable_checkpointing': True,
            'early_stop_patience_steps': 100,
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
    
    if 'save_top_k' not in config.keys():
        # Number of top checkpoints to keep; default preserves previous behavior
        config['save_top_k'] = 20

    # if 'lr_scheduler' not in config.keys():
    #     config['lr_scheduler'] = CosineAnnealingLR

    if 'optimizer' not in config.keys():
        config['optimizer'] = 'Adam'

    lr_scheduler = CosineAnnealingLR

        
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

    # if dataframe is provided, use it. Otherwise, initialize a new dataloader and find all continuous snippets
    if dataframe is not None:
        Y_df = dataframe
    else:
        print('--------------------------------')
        print('---Initializing dataloader...---')
        data = dataloader(config=config, train=True)
        Y_df = data.find_all_continuous_snippets(hours=config['snippet_length'])
        print('---Dataloader initialized-------')
        print('--------------------------------')
    
    horizon_hours = config['horizon'] # length of horizon in hours
    
    if overfit_size is not None:
        Y_df = Y_df[:overfit_size*32*config['snippet_length']]
        Y_df.to_csv('Y_df_overfit.csv')

    loss = MQLoss(level=[95, 97, 98])
    
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
                                             loss=loss,
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
                                             loss=loss,
                                             lr_scheduler=lr_scheduler,
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
                                             loss=loss,
                                             **trainer_kwargs
                                             )
                                )
    
    nf = NeuralForecast(
        models=models_array,
        freq=freq
    )
    
    val_size = config['horizon'] if padding else 0
    optimizer = Adam(nf.models[0].parameters())
    nf.models[0].lr_scheduler_kwargs = {"optimizer": optimizer, "T_max": config["max_steps"]/100}
    # create directory for saving model
    train_id_string = dt.datetime.strftime(dt.datetime.now(), '%Y_%m_%d_%H_%M_%S')+'_'+config['models']+'_'+str(config['snippet_length']-config['horizon'])+'_'+str(config['horizon'])+'_'+str(config['max_steps'])
    if config['enable_checkpointing']:
        os.makedirs(f'models/{train_id_string}', exist_ok=True)
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
    nf.models[0].valid_trajectories = nf.models[0].valid_trajectories[1:]
    
    try:
        train_loss = nf.models[0].train_trajectories
        val_loss = nf.models[0].valid_trajectories
            
        if train_loss is not None and val_loss is not None:
            plt.figure(figsize=(12, 8))
            
            # Calculate moving average window size (approximately one epoch)
            # Assuming validation checks happen every val_check_steps
            window_size = max(1, min(2500, len(train_loss) // 10))
            # different value for validation
            val_window_size = max(1, min(2500, len(val_loss) // 10))
            # window_size = 1 # for debugging
            
            # Apply moving average filter using numpy convolution
            # Convert to numpy arrays for easier manipulation
            train_array = np.array(train_loss)
            val_array = np.array(val_loss)

            for i, loss in enumerate(train_array):
                if loss[1] > 30:
                    try:
                        train_array[i][1] = train_array[i-1][1]
                    # erroneous_batch = Y_df[i*32*config['snippet_length']:(i+1)*32*config['snippet_length']-1]
                    # print(f'Erroneous batch at index {i}')
                    # print(erroneous_batch)
                    # right_batch = Y_df[(i-1)*32*config['snippet_length']:i*32*config['snippet_length']-i*32*config['snippet_length']-1]
                    # print(f'Right batch at index {i-1}')
                    # print(right_batch)
                    except:
                        pass
            
            # Apply moving average to y values using numpy convolution
            if len(train_loss) >= window_size:
                smoothed_train_y = np.convolve(train_array[:, 1], np.ones(window_size)/window_size, mode='valid')
                smoothed_train_x = train_array[window_size-1:, 0]
                smoothed_train_loss = np.column_stack((smoothed_train_x, smoothed_train_y))
            else:
                smoothed_train_loss = train_array
            
            if len(val_loss) >= val_window_size:
                smoothed_val_y = np.convolve(val_array[:, 1], np.ones(val_window_size)/val_window_size, mode='valid')
                smoothed_val_x = val_array[val_window_size-1:, 0]
                smoothed_val_loss = np.column_stack((smoothed_val_x, smoothed_val_y))
            else:
                smoothed_val_loss = val_array
            
            # Scale validation loss to match training loss span using MinMaxScaler
            scaler = MinMaxScaler()
            
            # Get the min and max of training loss for scaling reference
            train_min = np.min(smoothed_train_loss[:, 1])
            train_max = np.max(smoothed_train_loss[:, 1])
            
            # Scale validation loss to match training loss range
            val_scaled = scaler.fit_transform(smoothed_val_loss[:, 1].reshape(-1, 1)).flatten()
            val_scaled = val_scaled * (train_max - train_min) + train_min
            
            # Create scaled validation loss array
            smoothed_val_loss_scaled = np.column_stack((smoothed_val_loss[:, 0], val_scaled))
            
            # Convert to DataFrames for plotting
            df_smoothed_train = pd.DataFrame(smoothed_train_loss, columns=['x', 'y'])
            df_smoothed_val = pd.DataFrame(smoothed_val_loss_scaled, columns=['x', 'y'])
            
            # Plot smoothed training loss (bold)
            plt.plot(df_smoothed_train['x'], df_smoothed_train['y'], 
                    color='blue', label=f'Smoothed Training Loss (MA-{window_size})')
            
            # Plot scaled smoothed validation loss (bold)
            plt.plot(df_smoothed_val['x'], df_smoothed_val['y'], 
                    color='red', label=f'Scaled Smoothed Validation Loss (MA-{val_window_size})')
            
            plt.xlabel('Steps')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss (with Moving Average Filter)')
            plt.legend()
            plt.grid(True, alpha=0.3)
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
            
            # Workaround: Save to temp location first, then copy to models directory
            # This avoids file watchers or cleanup processes that seem to delete files in models subdirectories
            import tempfile
            import shutil
            
            try:
                # Save to a temporary file first
                temp_file = f'temp_losses_{train_id_string}.png'
                train_graph.savefig(temp_file, dpi=100, bbox_inches='tight')
                plt.close(train_graph)
                
                # Copy to the final destination (use outputs directory to avoid .gitignore)
                os.makedirs('outputs', exist_ok=True)
                final_path = f'outputs/losses_{train_id_string}.png'
                shutil.copy2(temp_file, final_path)
                
                # Clean up temp file
                os.remove(temp_file)
                
                # Verify final file exists
                if os.path.exists(final_path):
                    file_size = os.path.getsize(final_path)
                    print(f"  - Loss plot saved to: {final_path} (size: {file_size} bytes)")
                    
                else:
                    print(f"  - Error: Loss plot was not saved to {final_path}")
                    
            except Exception as e:
                print(f"  - Error saving loss plot: {e}")
                import traceback
                traceback.print_exc()
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
        best_model_path = f'models/{train_id_string}'

        # Construct the filename
        best_model_filename = f'best_model_{best_model_info["step"]:02d}_{best_model_info["value"]:.4f}.json'
        best_model_filepath = f'outputs/best_model_{train_id_string}.json'
        # Convert numpy types to native Python types for JSON serialization
        best_model_info_serializable = {
            'metric': best_model_info['metric'],
            'value': best_model_info['value'].item() if hasattr(best_model_info['value'], 'item') else best_model_info['value'],
            'step': best_model_info['step'].item() if hasattr(best_model_info['step'], 'item') else best_model_info['step'],
            'index': best_model_info['index'].item() if hasattr(best_model_info['index'], 'item') else best_model_info['index']
        }
        
        # Dump best_model_info to a JSON file
        with open(best_model_filepath, 'w') as f:
            json.dump(best_model_info_serializable, f, indent=4)
        print(f"  - Best model info saved to: {best_model_filepath}")
        
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


    # Move files from outputs/ to models/ directory after training is complete
    print("Moving files to models directory...")
    try:
        # Move loss plot if it exists
        loss_plot_src = f'outputs/losses_{train_id_string}.png'
        if os.path.exists(loss_plot_src):
            loss_plot_dst = f'models/{train_id_string}/losses.png'
            shutil.move(loss_plot_src, loss_plot_dst)
            print(f"  - Loss plot moved to: {loss_plot_dst}")
        
        # Move JSON file if it exists and best_model_info is available
        json_src = f'outputs/best_model_{train_id_string}.json'
        if os.path.exists(json_src) and best_model_info is not None:
            json_dst = f'models/{train_id_string}/best_model_{best_model_info["step"]:02d}_{best_model_info["value"]:.4f}.json'
            shutil.move(json_src, json_dst)
            print(f"  - Best model info moved to: {json_dst}")
        elif os.path.exists(json_src):
            # Fallback: move with generic name if best_model_info is not available
            json_dst = f'models/{train_id_string}/best_model_info.json'
            shutil.move(json_src, json_dst)
            print(f"  - Best model info moved to: {json_dst}")
            
    except Exception as e:
        print(f"  - Error moving files: {e}")

    print('\n\n\n')
    print("-----------------------------")
    print("--------Training done--------")
    print("-----------------------------")
    print('\n\n\n')

    return nf, train_graph

if __name__ == '__main__':
    _,  graph = run(debug=True, overfit_size=4)
    graph.savefig(fname='train_graph.png')
    