import flask
from flask import Flask, request
import json
import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast

nf = NeuralForecast.load(path='./TimesNet_21_3_20k')


def predict_with_timesnet( si_list_for_pred = []):
    SI_series = [6.80487369543902e-05, 6.54717434210467e-05, 0.0001413882726481, 0.0001610709799212,
                 6.8653942879885e-05, 0.0, 2.94545274914257e-05, 0.0001385052254881, 0.0002047841625041,
                 0.0002017716621774, 0.0001058899443597, 7.17093486762437e-05, 5.710913673916e-05, 0.0001280686598129,
                 0.0001967415532699, 7.56615780829805e-05, 7.14784876277932e-05, 0.0001094689010405, 0.0001419066720335,
                 9.12996175710565e-05, 8.02659378916849e-06]

    # function time
    y = pd.DataFrame(SI_series, columns=['y'])
    y['ds'] = pd.date_range(start='2020-01-01 00:00:00', periods=len(SI_series), freq='H')
    y['unique_id'] = 'uid_a'
    y_hat = nf.predict(y)
    # y_hat = pd.read_csv('in_silico/TimesNet_21_3_20k/test_out_single.csv')

    # residual time
    residuals = np.fromfile('in_silico/TimesNet_21_3_20k/residuals.txt')
    runs = 1000
    y_hat_out = y_hat.copy(deep=True)

    samples = []
    for _ in range(runs):
        sampled = y_hat_out[str(nf.models[0])] + np.random.choice(residuals,
                                                                  size=len(y_hat_out[str(nf.models[0])]),
                                                                  replace=True
                                                                  )
        samples.append(sampled)
    samples = np.stack(samples)
    lower = np.percentile(samples, 5, axis=0)
    lower = np.clip(lower, a_min=1e-7, a_max=None)
    upper = np.percentile(samples, 95, axis=0)
    # median = np.percentile(samples, 50, axis=0)

    y_hat_out['95'] = upper
    y_hat_out['05'] = lower
    # Y_hat_df_2_percentiles['median'] = median
    y_hat_out.drop(['Unnamed: 0', 'unique_id', 'ds', 'TimesNet'], axis=1, inplace=True)

    # output generation time
    pred = y_hat_out.iloc[:, [0, 1]].values.tolist()
    return []


app = Flask(__name__)

@app.route( '/', methods=['POST'] )
def index():

    content_type = request.headers.get( 'Content-Type' )

    data = request.get_json()

    si_list = data[ 'SI' ]
    insulin_list = data[ 'insulin' ]
    glucose_list = data[ 'glucose' ]
    bgt_list = data[ 'bgt' ]
    bg_list = data[ 'bg' ]
    age = data[ 'age' ]
    gender = data[ 'gender' ]

#     print( '---------------------' )
#     print( data )
    print( '---------------------' )
    print( f'SI = {si_list}' )                  # SI list
    print( f'Insulin = {insulin_list}' )        # administred insulin in every hour  
    print( f'Glucose = {glucose_list}' )        # administred glucose (enteral + all parenteral lines) in every hour 
    print( f'BGT = {bgt_list}' )                # measured BG time stamps
    print( f'BG = {bg_list}' )                  # corresponding measured BG values
    print( f'Age = {age}' )                     # age: years in double
    print( f'Gender = {gender}' )               # gender: 1 = male, -1 = female, 0 = no info about gender

    current_si = si_list[ -1 ]
    # print( f'current SI = {current_si}' )
    

    # empty array means no prediction, the controller will use the classic star prediction
    print(f'this is the SI list: {si_list}')
    if len(si_list) < 21:
        # legyen meg a korabbiakra is
        pred = []
    else:
        # timesnet prediction
        pred = predict_with_timesnet(si_list_for_pred=si_list[-21:])

    print()
    print(f'pred: {pred} ')

    resp = flask.Response( 
            json.dumps(
                    (pred[::-1]).tolist()
            )
    )

    resp.headers[ 'Content-Type' ] = 'application/json'

    return resp

