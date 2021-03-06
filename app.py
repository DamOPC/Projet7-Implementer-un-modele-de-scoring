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
data = 'api_sample.csv'
data_shap = 'df_shap_0.csv'
model = 'lgbm_test_model.sav'

# Variables
df = pd.read_csv(data, sep=',').drop('target', axis=1).sort_values(by='sk_id_curr')
df_shap = pd.read_csv(data_shap, sep=',').sort_values(by='sk_id_curr')
df_graph = pd.read_csv(data, sep=',')
estimator = pickle.load(open(model, 'rb'))

# Routes ping
@app.route("/ping", methods=["GET"])
def return_pong():
    return pong

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
    df_pred = df[df['sk_id_curr']==user]
    y_pred = estimator.predict_proba(df_pred)
    #Voir pour droper la clé
    zero_proba = y_pred[0,0]
    return json.dumps({'pred' : zero_proba})

# Routes Shap
@app.route("/shap", methods=["POST"])
def return_shap():
    user = json.loads(request.data)["ID"]
    df_sha = df_shap[df_shap['sk_id_curr']==user]
    df_sha = df_sha.drop('sk_id_curr', axis=1)
    df_sha_arr = df_sha.to_numpy()
    list_shap = df_sha_arr.tolist()
    json_shap = json.dumps(list_shap)
    return json_shap
 
# Routes DF
@app.route("/df", methods=["GET"])
def return_df():
    df_graph_json = df_graph.to_json()
    return json.dumps({'df_graph' : df_graph_json})

# Routes DF top10
@app.route("/dataframe", methods=["POST"])
def return_dataframe():
    cols = json.loads(request.data) 
    df_top = df[cols]
    df_top_json = df_top.to_json()
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
