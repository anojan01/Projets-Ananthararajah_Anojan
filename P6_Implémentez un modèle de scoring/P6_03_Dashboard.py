# streamlit run D:/OC/Projet_7/Livrables/P7_Anantharajah_Anojan/P7_03_Dashboard.py

import base64
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import pickle
import shap
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import japanize_matplotlib
import plotly.express as px


def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

# Import données & Modèle

X = pd.read_csv("data/X_train.csv")
X = X.head(500)
X = X[X.columns[1:]]
y = pd.read_csv("data/y_train.csv")
y = y[y.columns[1:]]
y = pd.DataFrame(y)

model = XGBClassifier()
model.load_model("mdl_final.json")

# FONCTIONS

# Collecter les caractéristiques du profil

st.sidebar.header("Les caractéristiques du client")


def client_carac():

    EXT_SOURCE_1 = st.sidebar.number_input('EXT_SOURCE_1',-100000.,100000.,0.)
    EXT_SOURCE_2 = st.sidebar.number_input('EXT_SOURCE_2',-100000.,100000.,0.)
    EXT_SOURCE_3 = st.sidebar.number_input('EXT_SOURCE_3',-100000.,100000.,0.)
    AMT_BALANCE = st.sidebar.number_input('AMT_BALANCE',-100000.,100000.,0.)
    AMT_CREDIT_MAX_OVERDUE = st.sidebar.number_input('AMT_CREDIT_MAX_OVERDUE',-100000.,100000.,0.)
    AMT_CREDIT_SUM_DEBT = st.sidebar.number_input('AMT_CREDIT_SUM_DEBT',-100000.,100000.,0.)
    AMT_REQ_CREDIT_BUREAU_TOTAL = st.sidebar.number_input('AMT_REQ_CREDIT_BUREAU_TOTAL',-100000.,100000.,0.)
    AMT_DOWN_PAYMENT = st.sidebar.number_input('AMT_DOWN_PAYMENT',-100000.,100000.,0.)
    AMT_CREDIT_SUM_OVERDUE = st.sidebar.number_input('AMT_CREDIT_SUM_OVERDUE',-100000.,100000.,0.)
    SK_DPD = st.sidebar.number_input('SK_DPD',-100000.,100000.,0.)
    REG_CITY_NOT_WORK_CITY = st.sidebar.number_input('REG_CITY_NOT_WORK_CITY',-100000.,100000.,0.)
    SELLERPLACE_AREA = st.sidebar.number_input('SELLERPLACE_AREA',-100000.,100000.,0.)
    FLAG_OWN_CAR = st.sidebar.number_input('FLAG_OWN_CAR',-100000.,100000.,0.)
    DAYS_LAST_DUE_1ST_VERSION = st.sidebar.number_input('BASEMENTAREA_MODE',0.,100.,0.)
    NFLAG_INSURED_ON_APPROVAL = st.sidebar.number_input('NFLAG_INSURED_ON_APPROVAL',-100000.,100000.,0.)
    FLAG_PHONE = st.sidebar.number_input('FLAG_PHONE',-100000.,100000.,0.)
    CODE_GENDER = st.sidebar.selectbox('CODE_GENDER : 0 Female / 1 Male',(0,1))
    ENTRANCES_MODE = st.sidebar.number_input('ENTRANCES_MODE',0.,100.,0.)
    CNT_INSTALMENT_MATURE_CUM = st.sidebar.number_input('CNT_INSTALMENT_MATURE_CUM',-100000.,100000.,0.)
    CREDIT_DAY_OVERDUE = st.sidebar.number_input('CREDIT_DAY_OVERDUE', -100000., 100000., 0.)

    data = {
        X.columns[0] : EXT_SOURCE_1,
        X.columns[1] : EXT_SOURCE_2,
        X.columns[2] : EXT_SOURCE_3,
        X.columns[3] : AMT_BALANCE,
        X.columns[4] : AMT_CREDIT_MAX_OVERDUE,
        X.columns[5] : AMT_CREDIT_SUM_DEBT,
        X.columns[6] : AMT_REQ_CREDIT_BUREAU_TOTAL,
        X.columns[7] : AMT_DOWN_PAYMENT,
        X.columns[8] : AMT_CREDIT_SUM_OVERDUE,
        X.columns[9] : SK_DPD,
        X.columns[10] : REG_CITY_NOT_WORK_CITY,
        X.columns[11] : SELLERPLACE_AREA,
        X.columns[12] : FLAG_OWN_CAR,
        X.columns[13] : DAYS_LAST_DUE_1ST_VERSION,
        X.columns[14] : NFLAG_INSURED_ON_APPROVAL,
        X.columns[15] : FLAG_PHONE,
        X.columns[16] : CODE_GENDER,
        X.columns[17] : ENTRANCES_MODE,
        X.columns[18] : CNT_INSTALMENT_MATURE_CUM,
        X.columns[19] : CREDIT_DAY_OVERDUE
     }

    profil = pd.DataFrame(data, index=[X.shape[0]])
    return profil

# Téléchargement du nouveau fichier csv

def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="X_file.csv">Download CSV File </a>'
    return href

# AFFICHAGE APPLICATION



# Affichage Accueil de l'application

img = "https://www.compareil.fr/public/medias/credit-scoring.jpeg"
st.image(img)

st.write("# Application permettant de prédire l'accord à un prêt bancaire")



# Affichage des données d'entrées

input_df = client_carac()
st.subheader("Synthèse Client")
st.write(input_df)



# Affichage de la prévision

prevision = model.predict_proba(input_df)[:,1]*100

st.subheader("Resultat de la prévision")

st.write(pd.DataFrame({"Probabilité d'être en défaut (Défault si >= 70%)": prevision}))

if prevision >= 70:
    st.subheader("Client en défault")
    y = y.append(pd.DataFrame({y.columns[0]: [1]},index=[y.shape[0]]))
else:
    st.subheader("Client conforme")
    y = y.append(pd.DataFrame({y.columns[0]: [0]},index=[y.shape[0]]))



# Ajout du client

X = pd.concat([X, input_df])
X_new = X

st.subheader("Synthèse globale des clients")
X_new["TARGET"] = y
st.write(X_new)



# Filtrage

st.subheader("Filtrage")

iter_lig = st.number_input("Nombre de lignes à afficher", 0, X.shape[0]-1, 2)

iter_col = st.number_input("Nombre de colonnes à afficher", 0, X.shape[1], 2)
col_list = []

for i in range(0, iter_col):
    index = st.selectbox("Colonne numéro {}".format(i + 1),X_new.columns,i)
    col_list.append(index)

X_filtre = X_new[col_list].iloc[:iter_lig,:]
st.write(X_filtre)

st.subheader("Téléchargement du fichier filtré")
st.markdown(filedownload(X_filtre), unsafe_allow_html=True)

# Affichage graphique

st.subheader("Graphiques")

for col in col_list:
    fig = px.scatter(
        x=X_new[col]
    )
    fig.update_layout(
        xaxis_title=col
    )
    st.write(fig)



# Affichage des interprétations via Shap

# Récupération du nombre de clients à afficher

st.subheader("Analyse Shap")

iter = st.number_input("Nombre d'individus à comparer", 0, X.shape[0] - 1, 2)

index_list = []

for i in range(0, iter):
    index = st.selectbox("Index Individu numéro {}".format(i + 1), range(0, X.shape[0]), i)
    index_list.append(index)


# Initialisation Shap

if st.button("Lancer l'analyse Shap") :

    shap.initjs()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    exp = shap.Explainer(model, X)
    val = exp.shap_values(X)

    # Shap Summary

    st.subheader("Shap Summary")
    shap.summary_plot(val, X)
    st.pyplot()

    # Shap Individual / Multi

    st.subheader("Shap Individual")

    for i in range(0,len(index_list)) :
        st.write("Interpretation Shap à l'index:",index_list[i],"- Prédiction :",X_new["TARGET"][index_list[i]])
        st_shap(shap.force_plot(exp.expected_value, val[index_list[i],:], X.iloc[index_list[i],:]))































































