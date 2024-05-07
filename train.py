import pandas as pd

from utils import to_datetime, connect_gt_and_pred

from statsforecast import StatsForecast
from neuralforecast import NeuralForecast
# from neuralforecast.models import NHITS, TimesNet, FEDformer, Informer, Autoformer
from neuralforecast.losses.pytorch import MQLoss

def run(config = None):
    if config['models'] == []:
        print("Error: no model configured!")
        return None, None
    
    Y_df = pd.read_csv("preliminary/full_dataset_hourly.csv")
    timestamps = Y_df['ds']
    new_timestamps = timestamps.apply(to_datetime)
    Y_df['ds'] = new_timestamps

    uids = Y_df['unique_id'].unique()
    Y_df = Y_df.query('unique_id in @uids').reset_index(drop=True)
    
    freq = 'h' # h or min
    horizon = 3 if freq=='h' else 90
    nf = NeuralForecast(
        models=[model(h=horizon, input_size=20, max_steps=1) for model in config['models']],
        freq=freq
    )

    df_cv = nf.cross_validation(Y_df, n_windows=1)

    df_to_plot = connect_gt_and_pred(df_gt=Y_df, df_pred=df_cv)

    plot = StatsForecast.plot(Y_df, df_to_plot.drop(columns=['y', 'cutoff']), max_insample_length=1260)

    print('\n\n\n')
    print("----------------------------")
    print("--------Training done-------")
    print("----------------------------")
    print('\n\n\n')

    return nf, plot

if __name__ == '__main__':
    run()
    