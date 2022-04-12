# Imports
from flask import Flask, jsonify, request
import pandas as pd
import json
import pickle
import requests
import urllib.request
import shap

# App config.
app = Flask(__name__)

# Datas
URL = 'https://raw.githubusercontent.com/DamOPC/Projet7/main'
data = URL + 'api_sample.csv'
model = URL + 'lgbm_test_model.sav'

# Variables
df = pd.read_csv(data, sep=',').drop('target', axis=1).sort_values(by='sk_id_curr')
estimator = pickle.load(urllib.request.urlopen(model))
#shap_values = 

# Routes features
@app.route("/features", methods=["GET"])
def return_features():
    features = list(df.columns).to_json()
    return features

# Routes clients IDs
@app.route("/ids", methods=["GET"])
def return_ids():
    ids = df['sk_id_curr']
    client_ids = ids.to_json()
    return client_ids

# Routes prediction
@app.route("/predict", methods=["POST"])
def predict(): 
    #DICT
    #response = data.text
    #user = response.get('ID')
    #print(user)
    #df_pred = df[df.sk_id_curr==user]
    #print(df_pred)
    #y_pred = estimator.predict_proba(df_pred)
    #JSON
    response = json.loads(request.data)
    user = response['ID']
    #print(user)
    #print(type(user))
    df_pred = df[df['sk_id_curr']==user]
    #df_pred = df[df.sk_id_curr==user]
    #print(df_pred)
    y_pred = estimator.predict_proba(df_pred)
    #print(type(y_pred))
    zero_proba = y_pred[0,0]
    return str(zero_proba)

# Routes Shap
@app.route("/shap", methods=["POST"])
def shap():
    return("status code:", bim)

#lancement de l'application
if __name__ == "__main__":
    print("Starting Python Flask server")
    app.run(debug=True)
