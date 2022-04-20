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
URL = 'https://raw.githubusercontent.com/DamOPC/Projet7/main/'
data = URL + 'api_sample.csv'
model = URL + 'lgbm_test_model.sav'
shap = URL + 'shap_values_0.p'

# Chargement des variables
df = pd.read_csv(data, sep=',').drop('target', axis=1).sort_values(by='sk_id_curr')
df_graph = pd.read_csv(data, sep=',')
estimator = pickle.load(urllib.request.urlopen(model))
shap_values = pickle.load(urllib.request.urlopen(shap))

# Routes features
@app.route("/features", methods=["GET"])
def return_features():
    features_list = list(df.columns)
    features = json.dumps(features_list)
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
    user = json.loads(request.data)["ID"]
    #print(user)
    #print(type(user)) 
    df_pred = df[df['sk_id_curr']==user]
    y_pred = estimator.predict_proba(df_pred)
    zero_proba = y_pred[0,0]
    #print(zero_proba)
    #print(type(zero_proba))
    return json.dumps({'pred' : zero_proba})

# Routes Shap
@app.route("/shap", methods=["POST"])
def return_shap():
    user = json.loads(request.data)["ID"]
    shap_value = shap_values[user]
    shap_list = shap_value.tolist()
    shap_json = json.dumps(shap_list) 
    return shap_json

# Routes DF
@app.route("/df", methods=["GET"])
def return_df():
    df_graph_json = df_graph.to_json()
    return json.dumps({'df_graph' : df_graph_json})

# Routes DF top10
@app.route("/dataframe", methods=["POST"])
def return_dataframe():
    cols = json.loads(request.data)
    print('cols type:', type(cols))    
    df_top = df[cols]
    print('df type:', type(df_top))
    print('df:', df_top)  
    df_top_json = df_top.to_json()
    print('json values type:', type(df_top_json))
    return json.dumps({'data' : df_top_json})

# Routes DF client
@app.route("/dataframeclient", methods=["POST"])
def return_dataframe_client():
    user = json.loads(request.data)["ID"]
    df_user = df[df['sk_id_curr']==user]
    df_user_json = df_user.to_json()
    return json.dumps({'dataUser' : df_user_json})

#lancement de l'application
if __name__ == "__main__":
    print("Starting Python Flask server")
    app.run(debug=True)
