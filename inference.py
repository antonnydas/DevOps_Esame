import logging
import json
import glob
import sys
from os import environ
from flask import Flask
from flask import request
from keras import models
import numpy as np

logging.debug('Init a Flask app')
app = Flask(__name__)


def doit(Pclass, Sex):
    MODEL_OUTPUT_PATH = os.environ['SM_MODEL_DIR']
    print(f"la model dir è {MODEL_OUTPUT_PATH}")
    model = models.load_model(f"{MODEL_OUTPUT_PATH}/Titanic_2_la_vendetta.h5")
    predict_input = np.array([
        [Pclass,Sex,0.125,0.5095,0.2165,0.1125,0.165,9]])
    predict_result = model.predict(predict_input)

    return json.dumps({"predict_result": predict_result.tolist()})

@app.route('/ping')
def ping():
    logging.debug('Hello from route /ping')
    Pclass = request.args.get("Pclass")
    Sex = request.args.get("Sex")
    

    return doit(float(Pclass), float(Sex))
