# %% imports

import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
from utils import dataloader, connect_gt_and_pred, calculate_error_distributions
from statsforecast import StatsForecast
from neuralforecast import NeuralForecast
from neuralforecast.losses.numpy import mae, mse, rmse
from metrics.old_model import OldModel as OM
from scipy.stats import norm

# %% config
config = {
        'normalize': True,
        'horizon': 3,
        'snippet_length': 6,
        'train_dataset': 'ICU_train',
        'test_dataset': 'ICU_test_10',
        'MQLoss_quantile': 90,
}

# model_path = 'models/2025_12_23_03_01_32_TimesNet_3_3_50000'
# model_path = 'models/2025_12_23_03_47_58_TimesNet_6_3_50000'
# model_path = 'models/2025_12_23_04_32_55_TimesNet_12_3_50000'
model_path = 'models/2025_12_23_05_14_57_TimesNet_21_3_50000'


horizon = config['horizon']
# snippet_length = int(model_path.split('_')[-3]) + int(model_path.split('_')[-2])
snippet_length = config['snippet_length']

# %% create results directory
results_path = model_path + '/results'
os.makedirs(results_path, exist_ok=True)

# %% load model
nf = NeuralForecast.load(path=model_path)
model_name = str(nf.models[0])

# %% load data
print('Loading dataset: ', config['test_dataset'])
data_test = dataloader(config=config, train=False)
Y_df_test = data_test.find_all_continuous_snippets(hours=snippet_length)
print('Done!')

# drop last horizon length y values for prediction
ids_to_drop = [i for i in range(len(Y_df_test)) if i % snippet_length >= snippet_length-horizon]
ids_to_drop_for_comparison = [i for i in range(len(Y_df_test)) if i % snippet_length < snippet_length-horizon]
Y_df_gt_window = Y_df_test.drop(ids_to_drop)
Y_df_gt_horizon = Y_df_test.drop(ids_to_drop_for_comparison)

input_window_length = snippet_length - horizon
chunk_size = 32 * input_window_length  # Process 32 complete windows at a time
if len(Y_df_gt_window) % chunk_size != 0:
    final_length = (len(Y_df_gt_window) // chunk_size) * chunk_size
    Y_df_gt_window_chunked = Y_df_gt_window.iloc[:final_length].reset_index(drop=True)
else:
    Y_df_gt_window_chunked = Y_df_gt_window

# %% predict
Y_hat_df_2 = nf.predict(df=Y_df_gt_window_chunked).reset_index()

# %% clip and denormalize predictions
for model in nf.models:
    Y_hat_df_2.rename(columns={f'{model}-median': str(model)}, inplace=True)
    Y_hat_df_2[str(model)] = Y_hat_df_2[str(model)].clip(lower=0.0)
    Y_hat_df_2[f'{model}-lo-95'] = Y_hat_df_2[f'{model}-lo-95'].clip(lower=0.0)
    Y_hat_df_2[f'{model}-hi-95'] = Y_hat_df_2[f'{model}-hi-95'].clip(lower=0.0)
    Y_hat_df_2[f'{model}-lo-97'] = Y_hat_df_2[f'{model}-lo-97'].clip(lower=0.0)
    Y_hat_df_2[f'{model}-hi-97'] = Y_hat_df_2[f'{model}-hi-97'].clip(lower=0.0)
    Y_hat_df_2[f'{model}-lo-98'] = Y_hat_df_2[f'{model}-lo-98'].clip(lower=0.0)
    Y_hat_df_2[f'{model}-hi-98'] = Y_hat_df_2[f'{model}-hi-98'].clip(lower=0.0)

if config['normalize']:

    # denormalize GT values
    Y_df_gt_horizon['y'] = data_test.scaler.inverse_transform(Y_df_gt_horizon['y'].values.reshape(-1,1))
    Y_df_test['y'] = data_test.scaler.inverse_transform(Y_df_test['y'].values.reshape(-1,1))
    
    # denormalize predicted values
    for model in nf.models:
        Y_hat_df_2[str(model)] = data_test.scaler.inverse_transform(Y_hat_df_2[str(model)].values.reshape(-1,1))
        Y_hat_df_2[f'{model}-lo-95'] = data_test.scaler.inverse_transform(Y_hat_df_2[f'{model}-lo-95'].values.reshape(-1,1))
        Y_hat_df_2[f'{model}-hi-95'] = data_test.scaler.inverse_transform(Y_hat_df_2[f'{model}-hi-95'].values.reshape(-1,1))
        Y_hat_df_2[f'{model}-lo-97'] = data_test.scaler.inverse_transform(Y_hat_df_2[f'{model}-lo-97'].values.reshape(-1,1))
        Y_hat_df_2[f'{model}-hi-97'] = data_test.scaler.inverse_transform(Y_hat_df_2[f'{model}-hi-97'].values.reshape(-1,1))
        Y_hat_df_2[f'{model}-lo-98'] = data_test.scaler.inverse_transform(Y_hat_df_2[f'{model}-lo-98'].values.reshape(-1,1))
        Y_hat_df_2[f'{model}-hi-98'] = data_test.scaler.inverse_transform(Y_hat_df_2[f'{model}-hi-98'].values.reshape(-1,1))

# %% plot trajectories
df_to_plot_2 = connect_gt_and_pred(df_gt=Y_df_test, df_pred=Y_hat_df_2, snippet_length=snippet_length, horizon=horizon)
prediction_graphs = StatsForecast.plot(
    Y_df_test,
    df_to_plot_2.drop(columns=['y', 'cutoff', 'level_0']),
    models=[f'{model_name}-lo-97',f'{model_name}-hi-97', f'{model_name}-lo-98', f'{model_name}-hi-98', f'{model_name}-lo-95', f'{model_name}-hi-95'],
    max_insample_length=1260,
)

# %% save trajectories
prediction_graphs.tight_layout()
prediction_graphs.savefig(fname=f'{results_path}/{model_name}_{snippet_length-horizon}_{horizon}_trajectories.png', bbox_inches='tight')


# %% prepare predictions for export and export
Y_hat_df_2_percentiles_merged = Y_hat_df_2.merge(
    Y_df_test[['unique_id', 'ds', 'y']],
    on=['unique_id', 'ds'],
    how='left'
)

cutoff_indices = [n * snippet_length + (snippet_length - horizon - 1) for n in range(len(Y_df_test) // snippet_length)]
cutoff_df = Y_df_test.iloc[cutoff_indices][['unique_id', 'ds', 'y']].rename(columns={'y': 'y_cutoff', 'ds': 'cutoff'}).reset_index(drop=True)

# Merge y_cutoff into Y_hat_df_2 based on unique_id and cutoff timestamp
Y_hat_df_2_percentiles_merged = Y_hat_df_2_percentiles_merged.merge(
    cutoff_df,
    on=['unique_id'],
    how='left'
)

Y_hat_df_2_percentiles_merged.to_csv(f'{results_path}/{model_name}_{snippet_length-horizon}_{horizon}_50k_all.csv', index=False)

# %% calculate outliers
quantiles = [95, 97, 98]
success_rates = []

for t in range(1, horizon+1):
    for quantile in quantiles:
        below_low = Y_hat_df_2_percentiles_merged['y'].astype(float)[t-1::horizon] < Y_hat_df_2_percentiles_merged[f'TimesNet-lo-{quantile}'].astype(float)[t-1::horizon] - 1e-7
        above_high = Y_hat_df_2_percentiles_merged['y'].astype(float)[t-1::horizon] > Y_hat_df_2_percentiles_merged[f'TimesNet-hi-{quantile}'].astype(float)[t-1::horizon] + 1e-7

        # Combine
        out_of_bounds = below_low | above_high

        # Count occurrences
        num_out_of_bounds = out_of_bounds.sum()

        success_rate = 1-(num_out_of_bounds/len(Y_hat_df_2))
        success_rates.append(success_rate)
        print(f'Success rate for {model_name} {snippet_length-horizon}->{horizon} with the {quantile}th percentile at t+{t} hour: {success_rate}')

# saving success rates
success_rates_df = pd.DataFrame({'quantile': quantiles, 'success_rate_t1': success_rates[0:horizon], 'success_rate_t2': success_rates[horizon:2*horizon], 'success_rate_t3': success_rates[2*horizon:3*horizon]})
success_rates_df.to_csv(f'{results_path}/{model_name}_{snippet_length-horizon}_{horizon}_success_rates.csv', index=False)


# %% plot success rates (grouped bars in one chart)
fig, ax = plt.subplots(figsize=(10, 6))

num_quantiles = len(quantiles)
num_t = horizon
x = np.arange(num_quantiles)
bar_width = 0.8 / max(num_t, 1)
colors = plt.cm.tab10.colors  # color cycle

# success_rates is t-major, then quantile: index = t * num_quantiles + q
for t in range(num_t):
    y_vals = [success_rates[t * num_quantiles + q] for q in range(num_quantiles)]
    offset = (t - (num_t - 1) / 2.0) * bar_width
    bars = ax.bar(x + offset, y_vals, width=bar_width, color=colors[t % len(colors)], label=f't+{t+1}h')

    # labels above bars
    for bar, sr in zip(bars, y_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{sr*100:.2f}%",
            ha='center', va='bottom', fontsize=9
        )


ax.set_xlabel('Quantile')
ax.set_ylabel('Success Rate')
ax.set_xticks(x, [f'{q}%' for q in quantiles])
ax.set_ylim(min(success_rates) - (max(success_rates) - min(success_rates)), 1)
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
ax.legend(title='Horizon')

fig.suptitle(f'Success Rates for {model_name} {snippet_length-horizon}->{horizon}')
fig.savefig(f'{results_path}/{model_name}_{snippet_length-horizon}_{horizon}_success_rates.png')

# %% Interval Ratio
# create interval ratios figures output directory within results path
interval_ratios_figures_path = results_path + '/interval_ratios_figures'
os.makedirs(interval_ratios_figures_path, exist_ok=True)
interval_ratios = []
for t in range(1, horizon+1):
    ref_model = OM(f'data/stochastic_{t}hr.csv')
    for quantile in quantiles:
        w_pred = Y_hat_df_2_percentiles_merged[f'{model_name}-hi-{quantile}'][t-1::horizon] - Y_hat_df_2_percentiles_merged[f'{model_name}-lo-{quantile}'][t-1::horizon]
        ref_ll, ref_hh = ref_model.predict(Y_hat_df_2_percentiles_merged['y_cutoff'][t-1::horizon])
        w_ref = ref_hh - ref_ll
        ir = w_pred / w_ref
        mir = np.mean(ir)
        print(f'Interval Ratio for {model_name} {snippet_length-horizon}->{horizon} with the {quantile}th percentile at t+{t} hour: {mir}')

        interval_ratios.append(mir)

        fig, ax = plt.subplots()
        ax.set_title(f'Interval Ratio for {model_name} {snippet_length-horizon}->{horizon} with the {quantile}th percentile at t+{t} hour')
        ax.set_xlim([0, 16]);
        ax.hist(ir, bins=np.linspace(0, 16, 100))
        fig.savefig(fname=f'{interval_ratios_figures_path}/{model_name}_{snippet_length-horizon}_{horizon}_interval_ratio_{quantile}th_percentile_{t}hr_horizon.png')

# %% save interval ratios

interval_ratios_df = pd.DataFrame({'quantile': quantiles, 'interval_ratio_t1': interval_ratios[0:horizon], 'interval_ratio_t2': interval_ratios[horizon:2*horizon], 'interval_ratio_t3': interval_ratios[2*horizon:3*horizon]})
interval_ratios_df.to_csv(f'{results_path}/{model_name}_{snippet_length-horizon}_{horizon}_interval_ratios.csv', index=False)

# %% calculate I-scores for each quantile
iscore_values = []
for t in range(1, horizon+1):
    for q, quantile in enumerate(quantiles):
        iscore = norm.ppf(0.5 + success_rates[q] / 2.0) / norm.ppf(quantile/100) / interval_ratios_df[f'interval_ratio_t{t}'][q]
        print(f'I-score for {model_name} {snippet_length-horizon}->{horizon} with the {quantile}th percentile at t+{t} hour: {iscore}')
        iscore_values.append(iscore)

iscore_values_df = pd.DataFrame({'quantile': quantiles, 'iscore_t1': iscore_values[0:horizon], 'iscore_t2': iscore_values[horizon:2*horizon], 'iscore_t3': iscore_values[2*horizon:3*horizon]})
iscore_values_df.to_csv(f'{results_path}/{model_name}_{snippet_length-horizon}_{horizon}_iscore.csv', index=False)

# %%
