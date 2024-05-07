import datetime as dt
import pandas as pd


def to_datetime(input_string):
    return dt.datetime.strptime(input_string, "%Y-%m-%d %H:%M:%S").replace(minute=0, second=0)


def connect_gt_and_pred(df_gt, df_pred):
    Y_df_2 = df_gt.set_index('ds')

    cv_df_2 = df_pred.reset_index()
    cutoff_stamps = []
    if 'cutoff' not in cv_df_2.columns:
        for i in range(len(cv_df_2)):
            hours_to_subtract = i%3 + 1
            cutoff_stamp = cv_df_2.at[i, 'ds'] - dt.timedelta(hours=hours_to_subtract, minutes=0)
            cutoff_stamps.append(cutoff_stamp)
        
        cv_df_2['cutoff'] = cutoff_stamps

    cv_df_output = pd.DataFrame()
    added_rows = 0
    for i in range(len(cv_df_2)):
        if i % 3 == 0:
            cv_df_output = pd.concat([cv_df_output, cv_df_2.iloc[i].to_frame().transpose(), cv_df_2.iloc[i:i + 3]], axis=0)
            cv_df_output.reset_index(drop=True, inplace=True)
            cv_df_output.at[i+added_rows, 'ds'] = cv_df_output.at[i+added_rows, 'cutoff']
            cutoff_timestamp = cv_df_2.iloc[i]['cutoff']
            unique_id = cv_df_2.iloc[i]['unique_id']
            
            while True:
                try:
                    gt_row = Y_df_2.loc[cutoff_timestamp]
                    break
                except:
                    cutoff_timestamp = cutoff_timestamp - dt.timedelta(hours=1, minutes=0)
                    
            if type(gt_row) == type(pd.DataFrame()):
                for j, row in enumerate(Y_df_2.loc[cutoff_timestamp].iterrows()):
                    if row[1]['unique_id'] == unique_id:
                        bg_gt = row[1]['y']
                        break
        
            else:
                bg_gt = Y_df_2.loc[cutoff_timestamp]['y']

            for col in cv_df_output.columns:
                if col not in ['unique_id', 'ds', 'cutoff']:
                    cv_df_output.at[i + added_rows, col] = bg_gt

            added_rows += 1

    cv_df_output.set_index('ds')

    if 'y' not in cv_df_2.columns:
        cv_df_output['y'] = cv_df_output['cutoff']

    if 'index' in cv_df_output.columns:
        cv_df_output.drop('index', axis=1, inplace=True)

    return cv_df_output
