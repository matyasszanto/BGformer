#!/bin/bash

. ../venv/bin/activate
# . ./in-silico/bin/activate

# python3 app.py
flask --app app run --host=0.0.0.0 --port=8008

