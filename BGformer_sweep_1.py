import wandb
import numpy as np
import train, test


# init weights and biases
wandb.login()


# experiment
def experiment():
    
    wandb.init()
    
    nf = train.run(config=wandb.config)
    prediction_plots, metrics, prediction_hists = test.run(config=wandb.config, models = nf)

    image_test = wandb.Image(prediction_plots, caption='Test plots')
    image_hist = wandb.Image(prediction_hists, caption='Error distributions')

    wandb.log(
        {
        'test plots': image_test,
        'metrics': metrics,
        'Error distributions': image_hist
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
        # datasets: ICU, Ohio
        'train_dataset': {
            'values': ['ICU'],
        },
        'test_dataset': {
            'values': ['Ohio'],
        },
        'normalize': {
            'values': [True]
        }
        # 'TN_topk': {
        #     'values': [2, 3, 4, 5, 6]
        # }
    }
}


# set up sweep and run it
sweep_id = wandb.sweep(sweep=config, project='BGformer_normalized')
wandb.agent(sweep_id=sweep_id, function=experiment)


# clean up
wandb.finish()
