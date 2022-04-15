# Modules
import requests
import json
import streamlit as st
import pandas as pd
import shap
import pickle
import streamlit.components.v1 as components
import numpy as np

# Viz
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image


if 'clicked' not in st.session_state:
    st.session_state['clicked'] = False
    
# Impression image (features globale)
threshold = 1 - 0.3

# Méthode pour afficher le force plot de shap
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

#Recup top colonnes
#def get_top_columns(shap_vals, index, f_names, num):
    #l = []
    # https://github.com/slundberg/shap/issues/632
    #for name in np.flip(np.argsort(np.abs(shap_vals[index]))[-num:]):
        #l.append(f_names[name])
    #return l    
    
#Shap values de la classe 0 
#expected_value = [-0.005649339905858291, 0.005649339905858291]
#shap_data = 'shap_values_0.p'
#shap_values = pickle.load(open(shap_data, 'rb'))

# Image features globales
#image_url = "https://banking-opc.herokuapp.com/images/dash3.png"
#image_url2 = "https://banking-opc.herokuapp.com/images/shap_importance.png"
#image_logo = Image.open(image_url)
#image_shap = Image.open(image_url2)
#st.image(image_logo)

#affichage formulaire
st.title('Dashboard Scoring Credit')
st.subheader("1. Prédictions de scoring client et comparaison à l'ensemble des clients")
#id_input = st.text_input('Veuillez saisir l\'identifiant d\'un client:', )
#chaine = "l'id Saisi est " + str(id_input)
#st.write(chaine)

client_IDs = requests.get(url="https://banking-opc.herokuapp.com/ids")
#client_IDs = requests.request(method='GET', url="https://banking-opc.herokuapp.com/ids")
#did = json.loads(client_IDs.text)
ID_dict = client_IDs.json()
id_input = st.selectbox('Selectionnez un ID client',ID_dict)
#IDs = list(ID_dict.values())
#id_input = st.selectbox('Selectionnez un ID client',IDs)

if st.button('Envoyez') or st.session_state.clicked: 
    st.session_state.clicked = True
    client_id = id_input
    #client_id = int(client_id)
    #print(client_id)
    try: 
        data_json = {'client_id': str(client_id)}
        response = requests.request(method='POST', headers={"Content-Type": "application/json"}, url="https://banking-opc.herokuapp.com/prediction", json=data_json)
        proba2 = float(response.json()["pred"])
        #pydict = {'ID': client_id}
        #jsondata = json.dumps(pydict)
        #response = requests.request(method='POST', url="https://banking-opc.herokuapp.com/predict", json=jsondata)
        #r = requests.post(url=URL, data=jsondata)
        #proba = float(response.json()['pred'])
    except:
        st.write("Wrong")
        response2 = requests.get(url="https://banking-opc.herokuapp.com/shap") 
        pred_dict = response2.json()
        pred_test = pred_dict['pred_test']
        st.write(pred_test)
        
        
    # Création jauge
    fig = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = proba2,    
    #value = proba,
    domain = {'x': [0, 1], 'y': [0, 1]},
    title = {'text': "Score"},
    gauge = { 'axis': {'range':[0,1]}}))

    if(proba > threshold):
        st.success("Le client a " + "{:.2%}".format(proba) + " de probabilité de remboursement.")
        fig.update_traces(gauge_bar_color="green")
    else:
        fig.update_traces(gauge_bar_color="red")
        st.error("Le client a " +"{:.2%}".format(1 - proba)+ " de probabilité de défaut de paiement !")

    # Affichage de la jauge
    st.write(fig)
    start = client_id
    #start = int(sample[sample["sk_id_curr"] == int(client_id)])
    
    
    st.subheader("2. Influence des variables sur le score du client– TOP 10")
    st.write("Conseil au chargé clientèle : Voici les variables ayant impacté la prédiction de solvabilité du client")
    #Force plot (features locales)
    #feature_names = dframe.columns.tolist()
    #feature_names = dframe.columns[~dframe.columns.isin(["sk_id_curr"])].tolist()
    #st_shap(shap.plots.force(expected_value[0], shap_values[start], feature_names))

    
    #waterfall (features locales)
    #fig, ax = plt.subplots()
    #shap.plots._waterfall.waterfall_legacy(expected_value[0], shap_values[start], feature_names=feature_names)
    #st.pyplot(fig, bbox_inches='tight',dpi=300,pad_inches=0)
    #plt.clf()
    
    st.subheader("3. Importance  des variables tout client confondu")
    st.write("Conseil au chargé clientèle : il peut être intéressant d'expliquer au client, quelles sont les variables qui influencent le plus pour l'obtention d'un prêt")
    # Impression image (features globale)
    #st.image(image_shap, caption='L\'importance de chaque caractéristique dans la décision')

    st.subheader("4. Situation du client sur les 10 principales variables")
    st.text("Le rond bleu représente la valeur du client sélectionné")
    # Récupérer les top 10 colonnes
    #top_cols = get_top_columns(shap_values, start, feature_names, 7)
    #fig, ax = plt.subplots()
    #ax.set_ylim([0, 2])
    #ax.set_xlim([0, 7])
    #fig.set_figwidth(30)
    #fig.set_figheight(20)
    #sns.boxplot(data=dframe[top_cols])
    #sns.stripplot(data=dframe[dframe["sk_id_curr"] == client_id][top_cols], color='blue',linewidth=1, size=20)
    #st.pyplot(fig, bbox_inches='tight',dpi=300,pad_inches=0)
    #lt.clf()

    st.subheader("5. Visualisation d'une ou plusieurs variables sélectionnées")
    # Liste déroulante
    #dfplot = pd.read_csv(r"C:\Users\Damien\Desktop\Data Scientist\P7\Dataset\modifie\essai3.csv")
    #feature_names = dfplot.columns[~dfplot.columns.isin(["sk_id_curr","target"])].tolist()
    #Variable = st.selectbox("feature_names 1:", feature_names) 
    #fig, ax = plt.subplots()
    #histplot = sns.histplot(data=dfplot, x=Variable, hue="target")
    #histplot.axvline(float(dfplot[dfplot["sk_id_curr"] == client_id][Variable]), color='red')
    #st.pyplot(fig, bbox_inches='tight',dpi=300,pad_inches=0)
    #plt.clf()

    # Liste déroulante
    #Variable2 = st.selectbox("feature_names 2:", feature_names) 
    #fig, ax = plt.subplots()
    #histplot2 = sns.histplot(data=dfplot, x=Variable2, hue="target")
    #histplot2.axvline(float(dfplot[dfplot["sk_id_curr"] == client_id][Variable2]), color='red')
    #st.pyplot(fig, bbox_inches='tight',dpi=300,pad_inches=0)
    #plt.clf()



