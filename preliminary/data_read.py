import pandas as pd
import matplotlib.pyplot as plt

P = 1

df = pd.read_csv(f"data/data_sip{P}.csv")

time_series_s = pd.DataFrame(columns=['id', 'values'])

for i, patient in enumerate(pd.unique(df["idx"])):

    # initialize empty list for SI values
    current_SI_vals = []

    # get subset of patient rows from full dataset
    df_current = df.loc[df['idx'] == patient]

    # append all SI values to list
    current_SI_vals.append(df_current.iloc[0]['SIM1'])
    current_SI_vals += [val for val in df_current['SI']]
    current_SI_vals.append(df_current.iloc[-1][f'SIP{P}'])
    time_series_s.at[i, 'id'] = patient
    time_series_s.at[i, 'values'] = current_SI_vals

# print(time_series_s.tail())

# plt.plot(time_series_s.at[5, 'values'])
# plt.show()

print(f'number of series before reduction: {len(time_series_s)}')

lens = []
max_len_index = 0
for i, time_series in enumerate(time_series_s.iterrows()):
    curr_len = len(time_series[1][1])
    if len(lens) > 0 and curr_len > max(lens):
        max_len_index = i
        longest = time_series
    if curr_len < 38:
        time_series_s.drop(index=i, inplace=True)
    else:
        lens.append(curr_len)
        numzeros = 0
        for el in time_series[1][1]:
            numzeros += 1 if el == 0 else 0
        if numzeros > curr_len / 15:
            time_series_s.drop(index=i, inplace=True)

longest_df = pd.DataFrame(longest)
print(longest_df.head())           


print(f'number of series after reduction: {len(time_series_s)}')
print(f'max = {max(lens)}')
print(f'min = {min(lens)}')

print(f'max_len_index: {max_len_index}')

print(time_series_s.head())