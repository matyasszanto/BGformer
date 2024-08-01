import pandas as pd
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
import datetime as dt

def find_all_continuous_snippets(dataset, hours=24):
    snippets = []
    total_points = len(dataset)
    snippet_counter = 0
    
    for start_idx in range(total_points - hours + 1):
        snippet = dataset.iloc[start_idx:start_idx + hours].copy()
        time_diffs = snippet['ds'].diff().iloc[1:]  # Skip the first NaT value
        if all(time_diffs == pd.Timedelta(hours=1)):
            snippet.loc[:, 'unique_id'] = f"{snippet['unique_id'].iloc[0]}_{snippet_counter}"
            snippets.append(snippet)
            snippet_counter += 1
    
    return pd.concat(snippets).reset_index(drop=True) if snippets else pd.DataFrame()


def icu_train_test_split(df, seed:int):
    np.random.seed(seed)
    ids = df['unique_id'].unique()
    test_idx = np.random.rand(len(ids)) > 0.8
    test_ids = ids[test_idx]
    train_ids = ids[~test_idx]
    df_test = df[df['unique_id'].isin(test_ids)]
    df_train = df[df['unique_id'].isin(train_ids)]

    df_train_random_snippets = find_all_continuous_snippets(df_train)
    df_test_random_snippets = find_all_continuous_snippets(df_test)

    return len(df_train_random_snippets), len(df_test_random_snippets), len(df_test_random_snippets)+len(df_train_random_snippets)


def to_datetime(input_string):
    return dt.datetime.strptime(input_string, "%Y-%m-%d %H:%M:%S").replace(minute=0, second=0)


dataframe = pd.read_csv('hourly_ICU_for_nf.csv')

timestamps = dataframe['ds']
new_timestamps = timestamps.apply(to_datetime)
dataframe['ds'] = new_timestamps

total_data_lens = []
for i in trange(0, 50):
    _, _, total = icu_train_test_split(dataframe, i)
    total_data_lens.append(total)

plt.plot(total_data_lens)
plt.savefig('out.png')
print(total_data_lens)
