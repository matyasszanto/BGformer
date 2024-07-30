import datetime as dt
import pandas as pd
import numpy as np
from tqdm import tqdm
import pathlib
from sklearn.preprocessing import MinMaxScaler

class dataloader():
    def __init__(self, config, train):

        self.config = config
        self.dataframe = self.read_dataset(train=train)
        self.min_bg = 0
        self.max_bg = 0
        self.normalize()
        

    def read_dataset(self, train):
        # read data
        train_key_selector = 'train_dataset' if train else 'test_dataset'
        basepath = pathlib.Path(__file__).parent.resolve()

        if self.config[train_key_selector] == 'Ohio':
            # OhioT1DM data
            csv_path = pathlib.Path.joinpath(basepath, "data/OhioT1DM/full_dataset_hourly.csv")
        else:
            # ICU data
            csv_path = pathlib.Path.joinpath(basepath, 'data/ICU_data/hourly_ICU_for_nf.csv')
 
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
        scaler = MinMaxScaler()
        self.dataframe['y'] = scaler.fit_transform(self.dataframe['y'].values.reshape(-1,1))


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


def connect_gt_and_pred(df_gt, df_pred, horizon=3):

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
    gt_uids = Y_df_2['unique_id'][::24].to_numpy()
    for i in tqdm(range(len(pred_df_2))):
        if i % horizon == 0:
            
            # fill cv_df_output with predictions and an extra row for t=0
            cv_df_output = pd.concat([cv_df_output, pred_df_2.iloc[i].to_frame().transpose(), pred_df_2.iloc[i:i + horizon]], axis=0)
            cv_df_output.reset_index(drop=True, inplace=True)
            cv_df_output.at[i+added_rows, 'ds'] = cv_df_output.at[i+added_rows, 'cutoff']
            
            # find current ground truth slice to merge
            pred_uid = pred_df_2.at[i, 'unique_id']
            current_gt_slice_index = np.where(gt_uids==pred_uid)[0].item()
            
            bg_gt = Y_df_2.loc[(current_gt_slice_index+1)*24-horizon-1]['y']

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
