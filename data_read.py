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

print(time_series_s.tail())

plt.plot(time_series_s.at[5, 'values'])
plt.show()
