import wandb
import train, test

debug = True

config = {
        # model types: NHITS, TimesNet, FEDformer, Informer, Autoformer, iTransformer
        'models': ["TimesNet"],
        # datasets: ICU, Ohio
        'train_dataset': 'ICU+Ohio',
        'test_dataset': 'ICU_test',
        'normalize': False,
        'TN_topk': 4,
        'max_steps': 1,
        'val_check_steps': 1000
    }
if not debug:
    wandb.init(
        entity='szanto-matyas',
        project='BGformer_normalized',
        config=config,
    )

nf = train.run(config=config)
prediction_plots, metrics, prediction_hists = test.run(config=config, models = nf)

if not debug:
    image_test = wandb.Image(prediction_plots, caption='Test plots')
    image_hist = wandb.Image(prediction_hists, caption='Error distributions')

    wandb.log(
        {
        #  'training plots': image_train,
        'test plots': image_test,
        'metrics': metrics,
        'Error distributions': image_hist
        }
    )

    wandb.finish()

else:
    prediction_hists.savefig(fname=f'testrun_out.png')
