import wandb
import numpy as np
import train, test

debug = True

config = {
        # model types: NHITS, TimesNet, FEDformer, Informer, Autoformer, iTransformer
        'models': ["TimesNet", "NHITS"],
        'TN_topk': 4,
    }
if not debug:
    wandb.init(
        entity='szanto-matyas',
        project='BGformer',
        config=config,
    )

nf = train.run(config=config)
plot_test, metrics = test.run(models = nf)

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
