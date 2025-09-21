import train, test

config = {
    "models": "TimesNet",
    'train_dataset': 'ICU_test_10',
    'test_dataset': 'ICU_test_10',
    'normalize': True,
    'max_steps': 1,
    'val_check_steps': 99,
    'horizon': 1,
    'snippet_length': 7,
    'TN_topk': 2,
    'enable_checkpointing': True,
    'early_stop_patience_steps': 5,
    'padding': True,
}

nf, losses_graph = train.run(config=config)
prediction_plots, metrics, prediction_hists, means_stds_df = test.run(config=config, models=nf)

prediction_plots.savefig(fname=f'testrun_out.png')