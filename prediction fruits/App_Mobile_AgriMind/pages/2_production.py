# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 14:53:47 2026

@author: Chris Landry
"""

import streamlit as st
import pandas as pd
from utils.config import CONFIG_PLANTES
from utils.models_loader import charger_model_rf
from utils.ui import show_header

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
import joblib

# INITIALISATION SECURISEE
if 'hist_temp' not in st.session_state:
    st.session_state['hist_temp'] = np.random.uniform(22,26,45).tolist()
if 'alertes' not in st.session_state:
    st.session_state['alertes'] = []

# Charger modèle RF
@st.cache_resource
def charger_model_rf():
    df = pd.read_csv("C:/Users/Marco/Desktop/Hackaton/IA/prediction fruits/App_Mobile_AgriMind/models/data/yield_df.csv")
    df['Area'] = df['Area'].str.strip()
    df['Item'] = df['Item'].str.strip()
    try:
        model = joblib.load("C:/Users/Marco/Desktop/Hackaton/IA/prediction fruits/App_Mobile_AgriMind/models/modele_maturation_agritech_final.pkl")
    except:
        features = ['Area','Item','avg_temp','average_rain_fall_mm_per_year']
        target = 'hg/ha_yield'
        preprocessor = ColumnTransformer([('cat',OneHotEncoder(handle_unknown='ignore'),['Area','Item'])], remainder='passthrough')
        model = Pipeline([('preprocessor',preprocessor), ('regressor', RandomForestRegressor(n_estimators=150, random_state=42))])
        model.fit(df[features],df[target])
        joblib.dump(model,"modele_maturation_agritech_final.pkl")
    return df, model

df_yield, model_yield = charger_model_rf()

# Config plantes
CONFIG_PLANTES = {
    'Cassava':{'temp_base':10,'seuil_floraison':1200,'seuil_maturite':2500,'nom':'Manioc','sensibilite':1.5},
    'Maize':{'temp_base':8,'seuil_floraison':850,'seuil_maturite':1600,'nom':'Maïs','sensibilite':1.2},
    'Potatoes':{'temp_base':5,'seuil_floraison':600,'seuil_maturite':1300,'nom':'Patate','sensibilite':1.0},
    'Rice, paddy':{'temp_base':10,'seuil_floraison':1100,'seuil_maturite':2000,'nom':'Riz','sensibilite':1.2},
    'Sorghum':{'temp_base':10,'seuil_floraison':900,'seuil_maturite':1800,'nom':'Sorgho','sensibilite':1.3},
    'Blueberry':{'temp_base':5,'seuil_floraison':300,'seuil_maturite':800,'nom':'Myrtille','sensibilite':1.4}
}

# Interface Production
st.header("🌾 Analyse de Rendement")

pays = st.selectbox("Pays", df_yield['Area'].unique())
plante = st.selectbox("Culture", list(CONFIG_PLANTES.keys()))
pluie = st.number_input("Pluie annuelle", value=1100)
temp_actuelle = st.number_input("Température actuelle (°C)", value=st.session_state['hist_temp'][-1])

def generer_rapport(pays,plante,temp_actuelle,pluie_actuelle):
    conf = CONFIG_PLANTES[plante]
    st.session_state['hist_temp'].append(temp_actuelle)
    historique_temp = st.session_state['hist_temp']
    gdd_total = sum(max(0,t-conf['temp_base']) for t in historique_temp)
    age = len(historique_temp)
    if gdd_total < conf['seuil_floraison']:
        stade = "Croissance végétative"; progression=(gdd_total/conf['seuil_floraison'])*50
    elif gdd_total < conf['seuil_maturite']:
        stade = "Floraison / Fructification"; progression=50+((gdd_total-conf['seuil_floraison'])/(conf['seuil_maturite']-conf['seuil_floraison']))*50
    else:
        stade = "Maturation finale"; progression=100

    stats = df_yield[(df_yield['Area']==pays)&(df_yield['Item']==plante)]
    temp_normale = stats['avg_temp'].mean() if not stats.empty else temp_actuelle
    rendement_moyen = stats['hg/ha_yield'].mean() if not stats.empty else 0
    input_data = pd.DataFrame([[pays,plante,temp_actuelle,pluie_actuelle]],columns=['Area','Item','avg_temp','average_rain_fall_mm_per_year'])
    pred_yield = model_yield.predict(input_data)[0]
    perte_pct = ((rendement_moyen - pred_yield)/max(rendement_moyen,1))*100
    perte_pct = np.clip(perte_pct,-40,40)
    seuil_alerte = 4.0 / conf['sensibilite']
    if temp_actuelle - temp_normale > seuil_alerte:
        statut = "🚨 ALERTE ROUGE"; conseil="IRRIGUER IMMÉDIATEMENT"
    elif temp_actuelle - temp_normale > seuil_alerte/2:
        statut = "⚠️ VIGILANCE"; conseil="ARROSAGE PRÉVENTIF"
    else:
        statut = "✅ ÉTAT STABLE"; conseil="Aucune action requise"

    if statut in ["🚨 ALERTE ROUGE","⚠️ VIGILANCE"]:
        st.session_state['alertes'].append({"Date":datetime.now().strftime("%Y-%m-%d %H:%M"), 
                                            "Pays":pays, "Plante":plante, "Temp":temp_actuelle, 
                                            "Statut":statut, "Action":conseil})
    # Ajouter l’alerte et afficher le toast automatiquement
    if statut in ["🚨 ALERTE ROUGE","⚠️ VIGILANCE"]:
        st.session_state['alertes'].append({
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Source": "Production",
            "Culture": plante,
            "Détail": f"Température critique : {temp_actuelle:.1f}°C",
            "Type": statut
        })
        st.toast(f"{statut} sur {plante} ! Conseil : {conseil}", icon="⚠️")
    
    rapport = f"""
🌱 CULTURE : {plante} ({conf['nom']})
📍 PAYS : {pays}
⏳ Age : {age} jours
🔥 GDD : {gdd_total:.1f}
Stade : {stade} | Progression : {progression:.1f}%
Temp actuelle : {temp_actuelle:.1f}°C | Temp normale : {temp_normale:.1f}°C
Impact rendement : {perte_pct:.1f} %
Conseil : {conseil}
Statut : {statut}
"""
    return rapport


df_yield, model = charger_model_rf()

if st.button("Générer rapport"):
    
    rapport = generer_rapport(pays,plante,temp_actuelle,pluie)
    st.code(rapport)

    input_data = pd.DataFrame(
        [[pays,plante,temp_actuelle,pluie]],
        columns=[
            'Area',
            'Item',
            'avg_temp',
            'average_rain_fall_mm_per_year'
        ]
    )

    pred = model.predict(input_data)[0]

    st.success(f"Rendement estimé : {pred:.2f}")