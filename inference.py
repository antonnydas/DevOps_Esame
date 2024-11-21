import os
import json
import boto3
import tarfile
import numpy as np
from flask import Flask, request
import tensorflow as tf

app = Flask(__name__)

def download_and_extract_model(remote_path, local_path):
    """
    Downloads and extracts the model from S3 to the local path.
    """
    s3 = boto3.client('s3')
    bucket, key = remote_path.replace("s3://", "").split("/", 1)

    if not os.path.exists(local_path):
        os.makedirs(local_path)
    
    local_tar_file = os.path.join(local_path, "model.tar.gz")
    
    if not os.path.exists(local_tar_file):  # Download only if not already downloaded
        print(f"Downloading model from {remote_path} to {local_tar_file}")
        s3.download_file(bucket, key, local_tar_file)
    else:
        print(f"Model archive already exists at {local_tar_file}")

    # Extract the tar file if it's not already extracted
    extracted_model_dir = os.path.join(local_path, "model")
    if not os.path.exists(extracted_model_dir):  # Extract only if not already extracted
        print(f"Extracting model to {extracted_model_dir}")
        with tarfile.open(local_tar_file, "r:gz") as tar:
            tar.extractall(path=extracted_model_dir)

    model_file = os.path.join(extracted_model_dir, "titanic_2_la_vendetta.h5")
    return model_file

def load_model():
    """
    Loads the model from the local directory.
    """
    model_path = os.getenv('MODEL_DIR', '/opt/ml/model')
    model_file = os.path.join(model_path, "Titanic_2_la_vendetta.h5")
    
    model = tf.keras.models.load_model(model_file)  # This works for both SavedModel and .h5
    
    return model

def doit(Pclass, Sex):
    # Define S3 path and local path for the model
    MODEL_S3_PATH = os.environ.get('SM_MODEL_DIR', 's3://bucket-prova333/output/tensorflow-training-2024-11-21-16-04-55-448/output/')
    LOCAL_MODEL_DIR = '/opt/ml/model'
    
    # Ensure the model is downloaded and extracted
    model_path = download_and_extract_model(MODEL_S3_PATH, LOCAL_MODEL_DIR)

    # Load the model
    model = load_model()

    # Prepare input and predict
    predict_input = np.array([[Pclass, Sex, 0.125, 0.5095, 0.2165, 0.1125, 0.165, 9]])
    predict_result = model.predict(predict_input)

    return json.dumps({"predict_result": predict_result.tolist()})

@app.route('/ping')
def ping():
    Pclass = request.args.get("Pclass")
    Sex = request.args.get("Sex")
    return doit(float(Pclass), float(Sex))

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)
