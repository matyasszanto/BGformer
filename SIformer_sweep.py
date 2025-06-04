import wandb
import numpy as np
import train, test


# init weights and biases
wandb.login()


# experiment
def experiment():
    
    wandb.init()
    
    nf = train.run(config=wandb.config)
    prediction_plots, metrics, prediction_hists, means_stds_df = test.run(config=wandb.config, models = nf)

    image_test = wandb.Image(prediction_plots, caption='Test plots')
    image_hist = wandb.Image(prediction_hists, caption='Error distributions')

    wandb.log(
        {
        'Predicted time series (test set)': image_test,
        'Error metrics': metrics,
        'best RMSE': min(metrics['RMSE']),
        'Error distributions': image_hist,
        'Error means and stds': means_stds_df,
        }
    )


# set up sweep config
config = {
    'method': 'grid',
    # 'run_cap': 5,
    'metric': {
        'goal': 'minimize',
        'name': 'best RMSE'
    },
    'parameters': {
        # model types: NHITS, TimesNet, FEDformer, Informer, Autoformer, iTransformer
        'models': {
            'values': ['TimesNet'],
            # 'values': ['TimesNet', 'FEDformer', 'Informer', 'Autoformer', 'iTransformer'],
        },
        # datasets: ICU, Ohio, ICU_test, ICU_train, ICU_test_10, ICU_valid_10, ICU+Ohio
        'train_dataset': {
            'values': ['ICU_train'],
        },
        'test_dataset': {
            'values': ['ICU_test_10'],
        },
        'normalize': {
            'values': [True]
        },
        'TN_topk': {
            'values': [2]
        },
        'val_check_steps': {
            'values': [21510]
        },
        'max_steps': {
            'values': [200000]
        },
        'horizon': {
            'values': [3]
        },
        'snippet_length': {
            'values': [24]
        },
        'enable_checkpointing': {
            'values': [True]
        }
    }
}


# set up sweep and run it
sweep_id = wandb.sweep(sweep=config, project='SIformer_models_train')
wandb.agent(sweep_id=sweep_id, function=experiment)


# clean up
wandb.finish()
