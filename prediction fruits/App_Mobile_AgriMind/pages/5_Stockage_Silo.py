import streamlit as st
import numpy as np
from datetime import datetime

st.set_page_config(page_title="🏗️ Stock IA", layout="wide")
if 'alertes' not in st.session_state:
    st.session_state['alertes'] = []

# --- CSS design agricole ---
st.markdown("""
<style>
h2 {color:white; text-align:center;}
body {background-color:#1b5e20;}
div.stButton > button {background-color:#2E7D32; color:white; border-radius:8px;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h2 style='text-align:center;'>🏗️ Analyse Stockage Silo et Alertes</h2>", unsafe_allow_html=True)

# --- Initialisation session_state ---
if 'alertes' not in st.session_state:
    st.session_state['alertes'] = []

if 'model_stockage' not in st.session_state:
    from sklearn.cluster import KMeans
    X = np.random.rand(200,3)
    model = KMeans(n_clusters=3,n_init=10, random_state=42)
    model.fit(X)
    st.session_state['model_stockage'] = model

model_stockage = st.session_state['model_stockage']

# --- Saisie capteurs ---
st.subheader("📊 Paramètres capteurs")
temp = st.slider("Température (°C)", 0, 60, 25)
hum = st.slider("Humidité (%)", 0, 100, 50)
co2 = st.slider("CO2 (ppm)", 0, 2000, 400)

# --- Analyse silo ---
if st.button("Analyser silo"):
    # Normalisation simple pour clustering
    data = np.array([[temp/60, hum/100, co2/2000]])
    cluster = model_stockage.predict(data)[0]

    if cluster == 0:
        st.success("✅ Stockage stable. Conseils : Maintenir conditions.")
    elif cluster == 1:
        st.warning("⚠️ Surveillance recommandée. Conseils : Ajuster ventilation / humidité.")
        st.toast("⚠️ Attention, stock instable !", icon="⚠️")
        st.session_state['alertes'].append({
            "Date": st.session_state.get('current_time', None) or "Non renseignée",
            "Type": "Stock Instable",
            "Temp": temp,
            "Hum": hum,
            "CO2": co2
        })
        st.session_state['alertes'].append({
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Source": "Stockage",
        "Culture": f"Silo {1}",
        "Détail": "Conditions instables",
        "Type": "⚠️ VIGILANCE"
        })
        st.toast("⚠️ Attention, stock instable !", icon="⚠️")
    else:
        st.error("🚨 Risque fermentation ! Vérifier température et CO2.")
        st.toast("🚨 Risque fermentation !", icon="⚠️")
        st.session_state['alertes'].append({
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Type": "🚨 Risque fermentation",
            "Temp": temp,
            "Hum": hum,
            "CO2": co2,
            "Source": "Stockage1",
            "Culture": f"Silo {1}",
            "Détail": "Conditions instables",
        })
        st.toast("🚨 Attention, stock mure pret pour distribution", icon="🚨")

    

