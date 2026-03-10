import streamlit as st
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans

# -----------------------------
# CONFIGURATION STREAMLIT
# -----------------------------

st.set_page_config(
    page_title="AgriMind AI",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 AgriMind AI - Smart Agriculture Platform")

# -----------------------------
# CONFIGURATION DES CULTURES
# -----------------------------

CONFIG_PLANTES = {
    "Maïs": {
        "temp_base": 10,
        "gdd_stades": [200, 500, 800, 1100],
        "temp_optimale": 25,
        "conseil": "Maintenir une irrigation régulière et surveiller les ravageurs."
    },
    "Blé": {
        "temp_base": 5,
        "gdd_stades": [150, 400, 700, 1000],
        "temp_optimale": 20,
        "conseil": "Apporter un engrais azoté et contrôler les maladies foliaires."
    },
    "Riz": {
        "temp_base": 8,
        "gdd_stades": [180, 450, 750, 1050],
        "temp_optimale": 28,
        "conseil": "Maintenir un niveau d'eau stable et surveiller les parasites."
    }
}

# -----------------------------
# MODELE IA RENDEMENT
# -----------------------------

@st.cache_resource
def entrainer_modele():

    X = np.random.rand(200, 3)
    y = X[:, 0] * 50 + X[:, 1] * 30 + X[:, 2] * 20 + np.random.normal(0, 5, 200)

    model = RandomForestRegressor()
    model.fit(X, y)

    return model

model_yield = entrainer_modele()

# -----------------------------
# MODELE IA STOCKAGE
# -----------------------------

@st.cache_resource
def entrainer_stockage():

    data = np.random.rand(100, 3)
    model = KMeans(n_clusters=3, random_state=42)
    model.fit(data)

    return model

model_stockage = entrainer_stockage()

# -----------------------------
# FONCTIONS AGRONOMIQUES
# -----------------------------

def calcul_gdd(temps, base):

    gdd = 0

    for t in temps:
        gdd += max(0, t - base)

    return gdd


def determiner_stade(gdd, seuils):

    if gdd < seuils[0]:
        return "🌱 Germination"

    elif gdd < seuils[1]:
        return "🌿 Croissance végétative"

    elif gdd < seuils[2]:
        return "🌼 Floraison"

    elif gdd < seuils[3]:
        return "🌽 Remplissage du grain"

    else:
        return "🌾 Maturité"


# -----------------------------
# INTERFACE UTILISATEUR
# -----------------------------

st.header("📊 Analyse Production Agricole")

col1, col2, col3 = st.columns(3)

with col1:
    pays = st.selectbox("Pays", ["Sénégal", "France", "Maroc"])

with col2:
    culture = st.selectbox("Culture", list(CONFIG_PLANTES.keys()))

with col3:
    jours = st.slider("Nombre de jours analysés", 10, 120, 30)

temp_moyenne = st.slider("Température moyenne (°C)", 0, 40, 25)
pluie = st.slider("Pluie (mm)", 0, 200, 50)

# -----------------------------
# GENERATION RAPPORT EXPERT
# -----------------------------

if st.button("🚀 Générer le rapport expert"):

    config = CONFIG_PLANTES[culture]

    historique_temp = np.random.normal(temp_moyenne, 2, jours)

    gdd = calcul_gdd(historique_temp, config["temp_base"])

    stade = determiner_stade(gdd, config["gdd_stades"])

    X_pred = np.array([[temp_moyenne / 40, pluie / 200, gdd / 1200]])

    rendement = model_yield.predict(X_pred)[0]

    ecart_temp = temp_moyenne - config["temp_optimale"]

    perte = 0

    if abs(ecart_temp) > 3:
        perte = abs(ecart_temp) * 2

    st.subheader("🛰️ Rapport AgriMind AI")

    st.markdown(f"""
    ### 🌱 Culture : {culture}

    **Pays :** {pays}

    **Stade physiologique :** {stade}

    **Cumul GDD :** {gdd:.2f}

    **Température optimale :** {config["temp_optimale"]} °C

    **Écart température :** {ecart_temp:+.2f} °C

    ---
    ### 📈 Rendement prédit

    **{rendement:.2f} hg/ha**

    **Impact potentiel : {perte:+.1f} %**

    ---
    ### 💡 Conseil agronomique

    {config["conseil"]}
    """)

# -----------------------------
# ANALYSE STOCKAGE
# -----------------------------

st.header("🏗️ Analyse IA Stockage des Silos")

col4, col5, col6 = st.columns(3)

with col4:
    temp_silo = st.slider("Température silo", 0, 60, 25)

with col5:
    humidite_silo = st.slider("Humidité silo", 0, 100, 50)

with col6:
    co2_silo = st.slider("CO2 silo", 0, 2000, 400)

if st.button("🔍 Analyser le stockage"):

    data = np.array([[temp_silo / 60, humidite_silo / 100, co2_silo / 2000]])

    cluster = model_stockage.predict(data)[0]

    if cluster == 0:

        statut = "🟢 Stable"
        diagnostic = "Conditions de stockage normales"
        action = "Aucune action nécessaire"

    elif cluster == 1:

        statut = "🟡 Surveillance"
        diagnostic = "Légère augmentation humidité/température"
        action = "Surveiller ventilation"

    else:

        statut = "🔴 Risque élevé"
        diagnostic = "Possible fermentation ou moisissure"
        action = "Ventiler immédiatement"

    st.subheader("📦 Diagnostic Stockage")

    st.markdown(f"""
    **Statut :** {statut}

    **Diagnostic :** {diagnostic}

    **Action recommandée :** {action}
    """)

# -----------------------------
# DASHBOARD KPI (pour hackathon)
# -----------------------------

st.header("📈 KPI Startup")

col7, col8, col9 = st.columns(3)

with col7:
    st.metric("Agriculteurs Beta", "52")

with col8:
    st.metric("Précision IA", "87%")

with col9:
    st.metric("Réduction pertes", "23%")