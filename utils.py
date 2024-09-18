import datetime as dt
import pandas as pd
import numpy as np
from tqdm import tqdm
import pathlib
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

class dataloader():
    def __init__(self, config, train, basepath=""):

        self.config = config
        self.basepath = basepath if basepath!="" else pathlib.Path(__file__).parent.resolve()
        self.dataframe = self.read_dataset(train=train)
        self.min_bg = 0
        self.max_bg = 0
        if config['normalize']:
            self.normalize()
        

    def read_dataset(self, train):
        # read data
        train_key_selector = 'train_dataset' if train else 'test_dataset'

        if self.config[train_key_selector] == 'Ohio':
            # OhioT1DM hourly train data
            csv_path = pathlib.Path.joinpath(self.basepath, "data/OhioT1DM/full_dataset_hourly.csv")
        elif self.config[train_key_selector] == 'ICU':
            # ICU data
            csv_path = pathlib.Path.joinpath(self.basepath, 'data/ICU_data/hourly_ICU_for_nf.csv')
        elif self.config[train_key_selector] == 'ICU_train':
            # ICU 80% split data
            csv_path = pathlib.Path.joinpath(self.basepath, 'data/ICU_data/hourly_ICU_train.csv')
        elif self.config[train_key_selector] == 'ICU_test':
            # ICU 20% split data
            csv_path = pathlib.Path.joinpath(self.basepath, 'data/ICU_data/hourly_ICU_test.csv')
        elif self.config[train_key_selector] == 'ICU+Ohio':
            # ICU 80% split data + Ohio data NORMALIZED!
            csv_path = pathlib.Path.joinpath(self.basepath, 'data/hourly_combo_train.csv')
        elif self.config[train_key_selector] == 'Ohio_test':
            # OhioT1DM hourly test data
            csv_path = pathlib.Path.joinpath(self.basepath, 'data/OhioT1DM/full_dataset_hourly_test.csv')
        else:
            print('Bad dataset selector option! Defaulting to Ohio dataset')
            csv_path = pathlib.Path.joinpath(self.basepath, "data/OhioT1DM/full_dataset_hourly.csv")
 
        Y_df = pd.read_csv(csv_path)
        
        if 'Unnamed: 0' in Y_df.columns:
            Y_df.drop(columns=['Unnamed: 0'], inplace=True)

        timestamps = Y_df['ds']
        new_timestamps = timestamps.apply(to_datetime)
        Y_df['ds'] = new_timestamps
        if 'bg' in Y_df.columns:
            Y_df.rename(columns={'bg': 'y'}, inplace=True)

        uids = Y_df['unique_id'].unique()
        Y_df = Y_df.query('unique_id in @uids').reset_index(drop=True)

        return Y_df
    

    def normalize(self):
        """
        save maximum and minimum BG values for later rescaling, and
        normalize BG values
        """
        self.min_bg = min(self.dataframe['y'])
        self.max_bg = max(self.dataframe['y'])
        self.scaler = MinMaxScaler()
        self.dataframe['y'] = self.scaler.fit_transform(self.dataframe['y'].values.reshape(-1,1))


    # Function to find all continuous snippets
    def find_all_continuous_snippets(self, hours=24):
        snippets = []
        total_points = len(self.dataframe)
        snippet_counter = 0
        
        for start_idx in range(total_points - hours + 1):
            snippet = self.dataframe.iloc[start_idx:start_idx + hours].copy()
            time_diffs = snippet['ds'].diff().iloc[1:]  # Skip the first NaT value
            if all(time_diffs == pd.Timedelta(hours=1)):
                snippet.loc[:, 'unique_id'] = f"{snippet['unique_id'].iloc[0]}_{snippet_counter}"
                snippets.append(snippet)
                snippet_counter += 1
        
        return pd.concat(snippets).reset_index(drop=True) if snippets else pd.DataFrame()


def to_datetime(input_string):
    return dt.datetime.strptime(input_string, "%Y-%m-%d %H:%M:%S").replace(minute=0, second=0)


def connect_gt_and_pred(df_gt, df_pred, snippet_length=24, horizon=3):

    # debug
    print('Started merging function')
    
    # local copies
    Y_df_2 = df_gt.reset_index()
    pred_df_2 = df_pred.reset_index()

    # create list of cutoff timestamps
    cutoff_stamps = []
    if 'cutoff' not in pred_df_2.columns:
        for i in range(len(pred_df_2)):
            hours_to_subtract = i%horizon + 1
            cutoff_stamp = pred_df_2.at[i, 'ds'] - dt.timedelta(hours=hours_to_subtract, minutes=0)
            cutoff_stamps.append(cutoff_stamp)
        
        pred_df_2['cutoff'] = cutoff_stamps

    print('Cutoff timestamps appended')

    cv_df_output = pd.DataFrame()
    added_rows = 0
    gt_uids = Y_df_2['unique_id'][::snippet_length].to_numpy()

    for i in tqdm(range(len(pred_df_2))):
        if i % horizon == 0:
            
            # fill cv_df_output with predictions and an extra row for t=0
            cv_df_output = pd.concat([cv_df_output, pred_df_2.iloc[i].to_frame().transpose(), pred_df_2.iloc[i:i + horizon]], axis=0)
            cv_df_output.reset_index(drop=True, inplace=True)
            cv_df_output.at[i+added_rows, 'ds'] = cv_df_output.at[i+added_rows, 'cutoff']
            
            # find current ground truth slice to merge
            pred_uid = pred_df_2.at[i, 'unique_id']
            current_gt_slice_index_array = np.where(gt_uids==pred_uid)[0]
            current_gt_slice_index = current_gt_slice_index_array.item()
            
            bg_gt = Y_df_2.loc[(current_gt_slice_index+1)*snippet_length-horizon-1]['y']

            for col in cv_df_output.columns:
                if col not in ['unique_id', 'ds', 'cutoff']:
                    cv_df_output.at[i + added_rows, col] = bg_gt

            added_rows += 1

    cv_df_output.set_index('ds')

    if 'y' not in pred_df_2.columns:
        cv_df_output['y'] = cv_df_output['cutoff']

    if 'index' in cv_df_output.columns:
        cv_df_output.drop('index', axis=1, inplace=True)

    return cv_df_output

def calculate_error_distributions(df_gt, df_pred, models, horizon=3):
    '''
    Function to calculate the distribution of error on the predicted horizon
    '''
    
    # get prediction model names
    if type(models) == str:
        model_strings = [models]
    else:
        model_strings = [str(model) for model in models.models]

    # initialize errors table and calculate errors for every model
    errors_array = np.empty(shape=(len(model_strings), horizon, int(len(df_gt)/horizon)))

    for i, [gt_row, pred_row] in enumerate(zip(df_gt.iterrows(), df_pred.iterrows())):
        gt_val = gt_row[1]['y']
        for j, model_string in enumerate(model_strings):
            error_val = (pred_row[1][model_string] - gt_val) / gt_val
            errors_array[j, int(i%horizon), int(i//horizon)] = error_val

    # calculate statistics
    means, stds = np.mean(errors_array, axis=2), np.std(errors_array, axis=2)

    # plot histograms
    fig, axes = plt.subplots(nrows=len(model_strings), ncols=horizon, figsize=(horizon*5, len(model_strings)*5))
    max_ylim = 0
    for i, ax in enumerate(axes.flat):
        model_indexer = i//horizon
        delta_t_indexer = i%horizon
        hist_arr = ax.hist(errors_array[model_indexer, delta_t_indexer, :], bins=200)
        plt.text(0, 1, f'Mean: {means[model_indexer, delta_t_indexer]:.4f}\nSTD: {stds[model_indexer, delta_t_indexer]:.4f}', ha='left', va='top', transform=ax.transAxes, bbox=dict(fill=True, facecolor='orange', edgecolor='black', linewidth=2))
        max_ylim = max_ylim if max(hist_arr[0])<max_ylim else max(hist_arr[0])
        
    for ax in axes.flat:
        ax.set_ylim(0, max_ylim + 0.15 * max_ylim)

    # mark rows and columns with models and time delays
    column_titles = [f't+{i+1}' for i in range(horizon)]
    if len(model_strings)>1:
        for ax, col in zip(axes[0], column_titles):
            ax.set_title(col)

        for ax, model_string in zip(axes[:, 0], model_strings):
            ax.annotate(model_string, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 5, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center')
    
    else:
        for ax, col in zip(axes, column_titles):
            ax.set_title(col)

        axes[0].annotate(model_strings[0], xy=(0, 0.5), xytext=(-axes[0].yaxis.labelpad - 5, 0), 
                         xycoords=axes[0].yaxis.label, textcoords='offset points',
                         size='large', ha='right', va='center')

    means_stds_array = np.empty(shape=(2*len(model_strings), horizon), dtype=means.dtype)
    means_stds_array[0::2] = means
    means_stds_array[1::2] = stds
    
    means_stds_df_indexer = pd.MultiIndex.from_product([model_strings, ['Mean', 'Std']], names=['Model name', 'Statistic'])
    
    means_stds_df = pd.DataFrame(means_stds_array, index=means_stds_df_indexer, columns=column_titles)
    means_stds_df.reset_index(inplace=True)

    print('\n\n\n')
    print(means_stds_df)

    return fig, means_stds_df
