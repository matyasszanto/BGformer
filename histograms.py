#%% Imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import title
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter

# Interactive window
plt.switch_backend('Qt5Agg')

def main(t_in=1, quantile_in=95):
    #%% Variables

    # ranges
    x_max = 1e-3
    y_max = 4e-3

    # histogram bins
    num_y_cutoff_bins = 100
    num_prediction_bins = 100

    # prediction distance
    t = t_in

    # horizon
    horizon = 3

    # quantile
    quantile = quantile_in

    model_name = 'TimesNet'

    #%% Setup
    # Load merged dataframe
    csv_file = f'{model_name}_12_3_50k_all.csv'
    merged = pd.read_csv(f'data/{csv_file}')
    train_window = csv_file.split('_')[-4]

    merged = merged[t::horizon]

    # Extract relevant columns as float
    x = merged['y_cutoff'].astype(float).values
    y = merged['y'].astype(float).values

    percentile_low = merged[f'TimesNet-lo-{quantile}'].astype(float).values
    percentile_high = merged[f'TimesNet-hi-{quantile}'].astype(float).values
    median = merged['TimesNet'].astype(float).values

    # Load Stochastic model values
    stoch = pd.read_csv(f'data/stochastic_{t}hr.csv')

    # Extract relevant stochastic model values as float
    stoch_x = stoch['x'].astype(float).values
    stoch_percentile_05 = stoch['05'].astype(float).values
    stoch_percentile_95 = stoch['95'].astype(float).values
    stoch_median = stoch['median'].astype(float).values


    #%% Filter to range

    # Predictions
    mask = (x >= 0) & (x <= x_max) & (y >= 0) & (y <= y_max)
    x = x[mask]
    y = y[mask]
    percentile_low= percentile_low[mask]
    percentile_high = percentile_high[mask]
    median = median[mask]

    stoch_mask = (stoch_x >= 0) & (stoch_x <= x_max)
    stoch_x = stoch_x[stoch_mask]
    stoch_percentile_05 = stoch_percentile_05[stoch_mask]
    stoch_percentile_95 = stoch_percentile_95[stoch_mask]
    stoch_median = stoch_median[stoch_mask]

    #%% Prepare bins
    y_cutoff_bins = np.linspace(0, x_max, num_y_cutoff_bins + 1)
    prediction_bins = np.linspace(0, y_max, num_prediction_bins + 1)
    prediction_bin_centers = 0.5 * (prediction_bins[:-1] + prediction_bins[1:])

    # Initialize Z matrix
    Z = np.zeros((num_y_cutoff_bins, num_prediction_bins))

    # Arrays for avg 5 and 95
    avg_5 = np.zeros(num_y_cutoff_bins)
    avg_95 = np.zeros(num_y_cutoff_bins)
    avg_median = np.zeros(num_y_cutoff_bins)

    # Arrays for avg stoch 5 and 95
    stoch_avg_5 = np.zeros(num_y_cutoff_bins)
    stoch_avg_95 = np.zeros(num_y_cutoff_bins)
    stoch_avg_median = np.zeros(num_y_cutoff_bins)

    #%% Loop over y_cutoff bins
    for i in range(num_y_cutoff_bins):
        bin_min = y_cutoff_bins[i]
        bin_max = y_cutoff_bins[i + 1]

        # Select rows in this y_cutoff bin
        bin_mask = (x >= bin_min) & (x < bin_max)
        y_in_bin = y[bin_mask]
        col_low_in_bin = percentile_low[bin_mask]
        col_high_in_bin = percentile_high[bin_mask]
        col_median_in_bin = median[bin_mask]

        # Select rows in this y bin for stoch
        stoch_bin_mask = (stoch_x >= bin_min) & (stoch_x < bin_max)
        stoch_col_5_in_bin = stoch_percentile_05[stoch_bin_mask]
        stoch_col_95_in_bin = stoch_percentile_95[stoch_bin_mask]
        stoch_col_median_in_bin = stoch_median[stoch_bin_mask]

        if len(y_in_bin) > 0:
            hist, _ = np.histogram(y_in_bin, bins=prediction_bins, density=False)
            Z[i, :] = hist
            avg_5[i] = np.mean(col_low_in_bin)
            avg_95[i] = np.mean(col_high_in_bin)
            avg_median[i] = np.mean(col_median_in_bin)
            stoch_avg_5[i] = np.mean(stoch_col_5_in_bin)
            stoch_avg_95[i] = np.mean(stoch_col_95_in_bin)
            stoch_avg_median[i] = np.mean(stoch_col_median_in_bin)
        else:
            Z[i, :] = 0
            avg_5[i] = np.nan  # No data
            avg_95[i] = np.nan
            avg_median[i] = np.nan
            stoch_avg_5[i] = np.nan
            stoch_avg_95[i] = np.nan
            stoch_avg_median[i] = np.nan

    #%% Optional: smooth Z with Gaussian filter
    Z_smooth = gaussian_filter(Z, sigma=1)

    #%% Prepare meshgrid for plotting
    y_cutoff_bin_centers = 0.5 * (y_cutoff_bins[:-1] + y_cutoff_bins[1:])
    X, Y = np.meshgrid(y_cutoff_bin_centers, prediction_bin_centers, indexing='ij')

    #%% Plot surface and lines
    fig = plt.figure(figsize=(12, 9))
    fig.canvas.manager.set_window_title(f'{model_name}, p(SI(t+{t})|t)')
    ax = fig.add_subplot(111, projection='3d')

    # Surface with transparency
    ax.plot_surface(X, Y, Z_smooth, cmap='viridis', alpha=0.5)

    # Plot avg 5 and 95 lines
    ax.plot(y_cutoff_bin_centers, avg_5, zs=0, zdir='z', color='red', linewidth=1, label=f'Average {model_name} {100-quantile}%')
    ax.plot(y_cutoff_bin_centers, avg_95, zs=0, zdir='z', color='blue', linewidth=1, label=f'Average {model_name} {quantile}%')
    ax.plot(y_cutoff_bin_centers, avg_median, zs=0, zdir='z', color='green', linewidth=1, label='Average median')
    ax.plot(y_cutoff_bin_centers, stoch_avg_5, zs=0, zdir='z', color='orange', linewidth=1, label='Stoch. model average 5%')
    ax.plot(y_cutoff_bin_centers, stoch_avg_95, zs=0, zdir='z', color='purple', linewidth=1, label='Stoch. model average 95%')
    ax.plot(y_cutoff_bin_centers, stoch_avg_median, zs=0, zdir='z', color='gray', linewidth=1, label='Stoch. model median%')

    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    ax.set_xlabel('SI at cutoff (GT)')
    ax.set_ylabel(f'Predicted SI at cutoff+{t}h')
    ax.set_zlabel('Count')
    ax.set_title(f't+{t} distribution surface + Average {100-quantile}% (red) & {quantile}% (blue)')

    ax.legend()

    # plt.tight_layout()
    # plt.show()

    #%% cones only
    # Plot top-down view with only the z=0 plane curves
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    ax2.plot(y_cutoff_bin_centers, avg_5, color='red', linewidth=1, label=f'Average {model_name} {100-quantile}%')
    ax2.plot(y_cutoff_bin_centers, avg_95, color='blue', linewidth=1, label=f'Average {model_name} {quantile}%')
    ax2.plot(y_cutoff_bin_centers, avg_median, color='green', linewidth=1, label='Average median')
    ax2.plot(y_cutoff_bin_centers, stoch_avg_5, color='orange', linewidth=1, label='Stoch. model average 5%')
    ax2.plot(y_cutoff_bin_centers, stoch_avg_95, color='purple', linewidth=1, label='Stoch. model average 95%')
    ax2.plot(y_cutoff_bin_centers, stoch_avg_median, color='gray', linewidth=1, label='Stoch. model median%')

    ax2.set_xlim(0, x_max)
    ax2.set_ylim(0, y_max)
    ax2.set_xlabel('SI at cutoff (GT)')
    ax2.set_ylabel(f'Predicted SI at cutoff+{t}h')
    ax2.set_title(f'Z=0 Top View: Curves for {model_name} & Stochastic Model')
    ax2.legend()
    # plt.tight_layout()
    plt.savefig(f'histograms/{model_name}_{train_window}_{horizon}_{quantile}_at_t{t}.png'.format(model_name), dpi=300)


if __name__ == "__main__":
    for t in [1, 2, 3]:
        for quantile in [95, 97, 98]:
            main(t, quantile)