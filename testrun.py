import wandb
import train, test

debug = False

config = {
        # model types: NHITS, TimesNet, FEDformer, Informer, Autoformer, iTransformer
        'models': ["TimesNet"],
        # datasets: ICU, Ohio
        'train_dataset': 'ICU_train',
        'test_dataset': 'ICU_test_10',
        'normalize': False,
        'TN_topk': 4,
        'max_steps': 1000,
        'val_check_steps': 500,
        'horizon': 3,
        'snippet_length': 24,
        'enable_checkpointing': True,
    }
if not debug:
    wandb.init(
        entity='szanto-matyas',
        project='SIformer_models_train',
        config=config,
    )

nf = train.run(config=config)
prediction_plots, metrics, prediction_hists, means_stds_df = test.run(config=config, models=nf)

if not debug:
    image_test = wandb.Image(prediction_plots, caption='Test plots')
    image_hist = wandb.Image(prediction_hists, caption='Error distributions')

    wandb.log(
        {
        #  'training plots': image_train,
        'Test plots': image_test,
        'Metrics': metrics,
        'Best RMSE': min(metrics['RMSE']),
        'Error distributions': image_hist,
        'Relative error means & stds': means_stds_df
        }
    )

    wandb.finish()

else:
    prediction_hists.savefig(fname=f'testrun_out.png')
