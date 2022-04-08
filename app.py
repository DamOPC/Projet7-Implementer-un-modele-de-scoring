# Imports
from flask import Flask, jsonify, request
import pandas as pd
import json
import pickle
import requests

# App config.
app = Flask(__name__)

# Datas/Variables
URL = "http://127.0.0.1:5000/"
dframe = pd.read_csv("api_sample.csv")
df = dframe.drop('target', axis=1)
model = 'lgbm_test_model.sav'
estimator = pickle.load(open(model, 'rb'))


# Routes test
@app.route("/", methods=["POST"])
def hello():
    return("<h1>Welcome!!<h1>", 200)

# Routes ping
@app.route("/ping", methods=["POST"])
def ping():
    response = requests.get(URL + "ping")
    return("status code:", response.status_code)

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
    app.run()
