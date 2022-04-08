# Imports
from flask import Flask, jsonify, request
import pandas as pd
import json
import pickle
import requests
import urllib.request

# App config.
app = Flask(__name__)

# Datas
URL = 'https://raw.githubusercontent.com/DamOPC/Projet7/main'
data = URL + 'api_sample.csv'
model = URL + 'lgbm_test_model.sav'

# Variables
dframe = pd.read_csv(data, sep=',')
df = dframe.drop('target', axis=1)
estimator = pickle.load(urllib.request.urlopen(model))

# Routes test
@app.route("/", methods=["POST"])
def hello():
    return("<h1>Welcome!!<h1>", 200)

# Routes clients IDs
@app.route("/sku", methods=["POST"])
def sku():
    return("status code:", bim)

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
