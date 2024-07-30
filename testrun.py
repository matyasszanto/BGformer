import wandb
import numpy as np
import train, test

debug = False

config = {
        # model types: NHITS, TimesNet, FEDformer, Informer, Autoformer, iTransformer
        'models': ["TimesNet", "NHITS"],
        # datasets: ICU, Ohio
        'train_dataset': 'ICU',
        'test_dataset': 'Ohio',
        'normalize': False,
        'TN_topk': 4,
    }
if not debug:
    wandb.init(
        entity='szanto-matyas',
        project='BGformer_normalized',
        config=config,
    )

nf = train.run(config=config)
plot_test, metrics = test.run(config=config, models = nf)

# image_train = wandb.Image(plot_train, caption=f'train plots')
if not debug:
    image_test = wandb.Image(plot_test, caption=f'test plots')

    wandb.log(
        {
        #  'training plots': image_train,
        'test plots': image_test,
        'metrics': metrics,
        }
    )

    wandb.finish()
