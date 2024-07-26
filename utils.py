import datetime as dt
import pandas as pd
import numpy as np
from tqdm import tqdm


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


# Function to find all continuous snippets
def find_all_continuous_snippets(df, hours=24):
    snippets = []
    total_points = len(df)
    snippet_counter = 0
    
    for start_idx in range(total_points - hours + 1):
        snippet = df.iloc[start_idx:start_idx + hours].copy()
        time_diffs = snippet['ds'].diff().iloc[1:]  # Skip the first NaT value
        if all(time_diffs == pd.Timedelta(hours=1)):
            snippet.loc[:, 'unique_id'] = f"{snippet['unique_id'].iloc[0]}_{snippet_counter}"
            snippets.append(snippet)
            snippet_counter += 1
    
    return pd.concat(snippets).reset_index(drop=True) if snippets else pd.DataFrame()