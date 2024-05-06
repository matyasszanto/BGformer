import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from neuralforecast.core import NeuralForecast
from neuralforecast.models import NHITS, NBEATS, TimesNet
from neuralforecast.losses.numpy import mae, mse

# df = pd.read_csv('D:\Developer\BGformer\data\OhioT1DM\glucose_value_540_training.csv')
# df['unique_id'] = 'OT'
# df['y'] = df['0']

# df['ds'] = pd.to_datetime(df['Unnamed: 0'])
# df = df.drop('Unnamed: 0', axis=1)
# df = df.dropna(axis=0)
df = pd.read_csv('D:\Developer\BGformer\data\OhioT1DM\\for_nf_glucose_value_540_training.csv')
df['unique_id'] = 'OT'
df['ds'] = pd.to_datetime(df['ds'])


# fig, ax = plt.subplots()

# ax.plot(df['y'])
# ax.set_xlabel('Time')
# ax.set_ylabel('BG')

# fig.autofmt_xdate()
# plt.tight_layout()
# plt.savefig("img_0.png")
# plt.show()

horizon=288

models = [NHITS(h=horizon,
               input_size=5*horizon,
               max_steps=50),
         NBEATS(h=horizon,
               input_size=5*horizon,
               max_steps=50),
         TimesNet(h=horizon,
                 input_size=5*horizon,
                 max_steps=50)
                 ]


nf = NeuralForecast(models=models, freq='min')
preds_df = nf.cross_validation(df=df, step_size=horizon, n_windows=1)

print(preds_df.head())

fig, ax = plt.subplots()

ax.plot(preds_df['y'], label='actual')
ax.plot(preds_df['NHITS'], label='N-HITS', ls='--')
ax.plot(preds_df['NBEATS'], label='N-BEATS', ls=':')
ax.plot(preds_df['TimesNet'], label='TimesNet', ls='-.')

ax.legend(loc='best')
ax.set_xlabel('Time steps')
ax.set_ylabel('BG')

fig.autofmt_xdate()
# plt.ylim(-0.001, 0.001)
plt.tight_layout()
plt.savefig('out_ohio.png')

data = {'N-HiTS': [mae(preds_df['NHITS'], preds_df['y']), mse(preds_df['NHITS'], preds_df['y'])],
       'N-BEATS': [mae(preds_df['NBEATS'], preds_df['y']), mse(preds_df['NBEATS'], preds_df['y'])],
       'TimesNet': [mae(preds_df['TimesNet'], preds_df['y']), mse(preds_df['TimesNet'], preds_df['y'])]}

metrics_df = pd.DataFrame(data=data)
metrics_df.index = ['mae', 'mse']

metrics_df.style.highlight_min(color='lightgreen', axis=1)
print(metrics_df)