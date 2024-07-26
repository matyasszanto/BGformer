import wandb
import numpy as np
import train, test


# init weights and biases
wandb.login()


# experiment
def experiment():
    
    wandb.init()
    
    nf = train.run(config=wandb.config)
    plot_test, metrics = test.run(models=nf)

    image_test = wandb.Image(plot_test, caption=f'test plots')

    wandb.log(
        {
        'test plots': image_test,
        'metrics': metrics,
        'best RMSE': min(metrics['RMSE'])
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
            'values': ['TimesNet', 'FEDformer'],
        },
        # 'TN_topk': {
        #     'values': [2, 3, 4, 5, 6]
        # }
    }
}


# set up sweep and run it
sweep_id = wandb.sweep(sweep=config, project='BGformer')
wandb.agent(sweep_id=sweep_id, function=experiment)


# clean up
wandb.finish()
