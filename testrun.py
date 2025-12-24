import train, test
from utils import dataloader

config = {
    "models": "TimesNet",
    'train_dataset': 'ICU_train_10',
    'test_dataset': 'ICU_test_10',
    'normalize': True,
    'max_steps': 2000,
    'val_check_steps': 100,
    'horizon': 3,
    'snippet_length': 24,
    'TN_topk': 2,
    'enable_checkpointing': True,
    'early_stop_patience_steps': 1000,
    'padding': True,
}

data = dataloader(config=config, train=True)
Y_df = data.find_all_continuous_snippets(hours=config['snippet_length'])

# for i in range(1, 6):
#     print('\n\n\n')
#     print("-----------------------------")
#     print("---Running overfitting test--")
#     print(f'---Overfitting size: {i}-------')
#     print("-----------------------------")
#     print('\n\n\n')

nf, losses_graph = train.run(config=config, overfit_size=1, dataframe=Y_df)
prediction_plots, metrics, prediction_hists, means_stds_df = test.run(config=config, models=nf)

prediction_plots.savefig(fname=f'testrun_out.png')
# losses_graph.savefig(fname=f'testrun_losses.png')