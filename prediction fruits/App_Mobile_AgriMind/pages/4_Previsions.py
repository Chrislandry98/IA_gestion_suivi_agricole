import streamlit as st
import numpy as np
from datetime import datetime

st.set_page_config(page_title="📡 Prévisions 32 jours", layout="wide")

# --- CSS design agricole ---
st.markdown("""
<style>
h2 {color:white; text-align:center;}
body {background-color:#1b5e20;}
div.stButton > button {background-color:#2E7D32; color:white; border-radius:8px;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h2 style='text-align:center;'>📡 Prévision Température et Pluviométrie sur 32 jours</h2>", unsafe_allow_html=True)

# --- Initialisation session_state ---
if 'hist_temp' not in st.session_state:
    st.session_state['hist_temp'] = np.random.uniform(22,26,45).tolist()
if 'alertes' not in st.session_state:
    st.session_state['alertes'] = []

# --- Configuration plantes ---
CONFIG_PLANTES = {
    'Cassava':{'temp_base':10,'seuil_floraison':1200,'seuil_maturite':2500,'sensibilite':1.5},
    'Maize':{'temp_base':8,'seuil_floraison':850,'seuil_maturite':1600,'sensibilite':1.2},
    'Potatoes':{'temp_base':5,'seuil_floraison':600,'seuil_maturite':1300,'sensibilite':1.0},
    'Rice, paddy':{'temp_base':10,'seuil_floraison':1100,'seuil_maturite':2000,'sensibilite':1.2},
    'Sorghum':{'temp_base':10,'seuil_floraison':900,'seuil_maturite':1800,'sensibilite':1.3},
    'Blueberry':{'temp_base':5,'seuil_floraison':300,'seuil_maturite':800,'sensibilite':1.4}
}

# --- Sélection de la culture ---
plante = st.selectbox("Culture", list(CONFIG_PLANTES.keys()))

# --- Générer prévisions aléatoires ---
future_temp = np.random.uniform(22,35,32)
future_pluie = np.random.uniform(0,20,32)

st.subheader("📈 Température prévue (32j)")
st.line_chart(future_temp)

st.subheader("🌧️ Pluviométrie prévue (32j)")
st.line_chart(future_pluie)

# --- Analyse maturation ---
gdd_pred = sum(max(0,t-CONFIG_PLANTES[plante]['temp_base']) for t in st.session_state['hist_temp'] + future_temp.tolist())
conf = CONFIG_PLANTES[plante]

if gdd_pred > conf['seuil_maturite']:
    st.success("✅ La culture sera mature dans les prochains jours. Récolte possible.")
elif gdd_pred > conf['seuil_floraison']:
    st.info("⚠️ Phase de floraison / fructification. Surveiller arrosage.")
else:
    st.warning("🌱 Croissance végétative. Prévoir irrigation si nécessaire.")

# --- Conseils détaillés pour les 3 premiers jours ---
st.subheader("Prévision détaillée 3 prochains jours")
for i,(t,r) in enumerate(zip(future_temp[:3],future_pluie[:3])):
    conseils = []
    if t>30: conseils.append("🔴 Temp élevée : Irrigation recommandée")
    if t<15: conseils.append("⚠️ Temp basse : Surveillance gel")
    if r<5: conseils.append("💧 Pluie faible : Irrigation préventive")
    if r>15: conseils.append("🌧️ Pluie forte : Drainage / protection sol")
    conseils_txt = ", ".join(conseils) if conseils else "Conditions normales"
    st.markdown(f"<div style='background-color:#F0F8FF; padding:8px; border-radius:5px; margin:3px 0;'><b>Jour {i+1}</b> - Temp: {t:.1f}°C, Pluie: {r:.1f} mm → Conseils: {conseils_txt}</div>", unsafe_allow_html=True)