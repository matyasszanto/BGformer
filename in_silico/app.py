import os

import flask
from flask import Flask, request
import json
import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast
import pytorch_lightning as pl


def load_model(horizon=3, window=3):
    pattern = f"{window}_{horizon}"
    for subdirectory in os.listdir(f'./models/horizon_{horizon}'):
        full_path = os.path.join('.', subdirectory)
        if os.path.isdir(full_path) and pattern in subdirectory:
            print(full_path)
            return NeuralForecast.load(full_path)
    return None


def predict_with_timesnet( si_list_for_pred = [], high_percent = 95):

    # function time
    y = pd.DataFrame(si_list_for_pred, columns=['y'])
    y['ds'] = pd.date_range(start='2020-01-01 00:00:00', periods=len(si_list_for_pred), freq='H')
    y['unique_id'] = 'uid_a'

    original_init = pl.Trainer.__init__

    def patched_init(self, *args, **kwargs):
        kwargs['logger'] = False
        kwargs['enable_checkpointing'] = False
        return original_init(self, *args, **kwargs)

    pl.Trainer.__init__ = patched_init


    # prediction time
    y_hat = nf.predict(y)

    high_key = f'TimesNet-hi-0.{high_percent}'
    # output generation time
    pred1 = [y_hat[high_key][0], y_hat['TimesNet-lo-0.95'][0]]
    if horizon == 3:
        pred2 = [y_hat[high_key][1], y_hat['TimesNet-lo-0.95'][1]]
        pred3 = [y_hat[high_key][2], y_hat['TimesNet-lo-0.95'][2]]

    pred = np.concatenate([pred1]) if window == 1 else np.concatenate([pred1, pred2, pred3])

    return pred

horizon = 3
window = 3
high_percent = 97
nf = load_model(horizon=horizon, window=window)

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
    

    # empty array means no prediction, the controller will use the classic star prediction
    print(f'this is the SI list: {si_list}')
    if len(si_list) < window:
        # legyen meg a korabbiakra is
        pred = []
    else:
        # timesnet prediction
        pred = predict_with_timesnet(si_list_for_pred=si_list[-window:], high_percent=high_percent)

    print()
    print(f'pred: {pred} ')

    resp = flask.Response( 
            json.dumps(
                    (pred[::-1]).tolist()
            )
    )

    resp.headers[ 'Content-Type' ] = 'application/json'

    return resp

