# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 21:15:16 2026

@author: Chris Landry
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- CONFIGURATION PAGE ---
st.set_page_config(page_title="Agri-Tech MVP", page_icon="🌾")

# --- CHARGEMENT DES MODÈLES ---
@st.cache_resource
def load_models():
    prod_model = joblib.load('rf_model.pkl')
    stock_model = joblib.load('kmeans_model.pkl')
    scaler = joblib.load('scaler_km.pkl')
    return prod_model, stock_model, scaler

try:
    rf, km, scaler = load_models()
except:
    st.error("⚠️ Modèles .pkl non trouvés. Assure-toi de les avoir générés avant.")

# --- BARRE LATÉRALE (NAVIGATION) ---
service = st.sidebar.radio("Choisir un Service", ["🌾 Suivi Production", "🏗️ Stockage IoT"])

# ==========================================
# SERVICE 1 : PRODUCTION (RANDOM FOREST)
# ==========================================
if service == "🌾 Suivi Production":
    st.header("🌾 Suivi de la Production")
    
    col1, col2 = st.columns(2)
    with col1:
        pays = st.selectbox("Pays", ["Angola", "Côte d'Ivoire", "Sénégal"])
        plante = st.selectbox("Culture", ["Cassava", "Maize", "Potatoes"])
    with col2:
        temp = st.slider("Température actuelle (°C)", 15.0, 45.0, 28.0)
        pluie = st.number_input("Pluie annuelle (mm)", value=1100)

    if st.button("Analyser la récolte"):
        # Simulation GDD & Progression
        progression = 28.9 # Valeur statique pour la démo
        st.write(f"**Stade actuel :** Croissance végétative")
        st.progress(progression / 100)
        
        # Prédiction IA
        input_data = pd.DataFrame([[pays, plante, temp, pluie]], 
                                  columns=['Area', 'Item', 'avg_temp', 'average_rain_fall_mm_per_year'])
        rendement = rf.predict(input_data)[0]
        
        st.metric("Rendement Estimé", f"{rendement:.2f} hg/ha", delta="-8.6%")
        st.info("💡 **Conseil :** Maintenir une humidité stable pour le développement des tubercules.")

# ==========================================
# SERVICE 2 : STOCKAGE (K-MEANS)
# ==========================================
if service == "🏗️ Stockage IoT":
    st.header("🏗️ Surveillance Silo Temps Réel")
    
    st.write("Simulateur de capteurs IoT :")
    c1, c2, c3 = st.columns(3)
    s_silo = c1.number_input("Temp. Silo (°C)", value=118.0)
    s_grain = c2.number_input("Temp. Grain (°C)", value=48.5)
    s_ambiant = c3.number_input("Temp. Ambiant (°C)", value=-2.5)

    if st.button("Vérifier l'état du Silo"):
        # Prétraitement et Prédiction
        features = np.array([[s_silo, s_grain, s_ambiant]])
        features_scaled = scaler.transform(features)
        cluster = km.predict(features_scaled)[0]
        
        # Logique d'affichage
        if cluster == 1: # Critique
            st.error("🚨 STATUT : CRITIQUE")
            st.warning("🔍 Diagnostic : Surchauffe/Fermentation détectée !")
            st.button("⚡ ACTIVER VENTILATION FORCÉE")
        elif cluster == 2: # Vigilance
            st.warning("⚠️ STATUT : VIGILANCE")
            st.info("🔍 Diagnostic : Instabilité thermique ou humidité suspecte.")
            st.write("📌 **Action :** Inspection manuelle requise.")
        else: # Optimal
            st.success("✅ STATUT : OPTIMAL")
            st.write("🔍 Diagnostic : Conservation stable.")
