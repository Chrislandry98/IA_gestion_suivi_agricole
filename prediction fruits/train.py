import streamlit as st
import pandas as pd
import numpy as np
import os
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# =================================================================
# 1. CONFIGURATION & ÉTAT DE LA SESSION
# =================================================================
st.set_page_config(page_title="AgriMind Global Expert", layout="wide")

if 'hist_temp' not in st.session_state:
    st.session_state['hist_temp'] = np.random.uniform(22, 28, 45).tolist()

if 'expeditions' not in st.session_state:
    st.session_state['expeditions'] = []

# =================================================================
# 2. MOTEURS IA
# =================================================================
@st.cache_resource
def charger_ia():
    base_path = os.path.dirname(__file__)
    prod_path = os.path.join(base_path, 'yield_df.csv')
    stock_path = os.path.join(base_path, 'archive/synthetic_industrial_data_with_status.csv')

    try:
        df_p = pd.read_csv(prod_path).dropna()
        df_p['Area'] = df_p['Area'].str.strip()
        df_p['Item'] = df_p['Item'].str.strip()
        
        model_p = Pipeline(steps=[
            ('preprocessor', ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), ['Area', 'Item'])], remainder='passthrough')),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        model_p.fit(df_p[['Area', 'Item', 'avg_temp', 'average_rain_fall_mm_per_year']], df_p['hg/ha_yield'])
        
        df_s = pd.read_csv(stock_path).dropna(subset=['sensor_3', 'sensor_11', 'sensor_18'])
        sc = StandardScaler()
        km = KMeans(n_clusters=3, random_state=42, n_init=10).fit(sc.fit_transform(df_s[['sensor_3', 'sensor_11', 'sensor_18']]))
        
        return df_p, model_p, km, sc
    except Exception as e:
        st.error(f"Erreur : {e}")
        st.stop()

df_yield, model_prod, model_stock, scaler_stock = charger_ia()

CONFIG_PLANTES = {
    'Cassava': {'temp_base': 10.0, 'seuil_flo': 1200, 'seuil_mat': 2500, 'nom': 'Manioc', 'conseil': "Stabilité hydrique requise."},
    'Maize': {'temp_base': 8.0, 'seuil_flo': 850, 'seuil_mat': 1600, 'nom': 'Maïs', 'conseil': "Surveiller la pollinisation."},
    'Rice, paddy': {'temp_base': 10.0, 'seuil_flo': 1100, 'seuil_mat': 2000, 'nom': 'Riz', 'conseil': "Niveau d'eau constant."}
}

# =================================================================
# 3. INTERFACE
# =================================================================
tabs = st.tabs(["🌾 PRODUCTION", "📡 PROD LIVE", "🏗️ STOCK IA", "📟 STOCK LIVE", "🚛 DISTRIBUTION & CLIENTS", "📈 PRÉVISIONS", "📞 CONTACTS"])

# --- 1. PRODUCTION ---
with tabs[0]:
    st.header("Analyse Agronomique Expert")
    col1, col2 = st.columns([1, 2])
    with col1:
        pays = st.selectbox("Pays", sorted(df_yield['Area'].unique()))
        plante = st.selectbox("Culture", list(CONFIG_PLANTES.keys()))
        t_site = st.slider("Température Site (°C)", 10.0, 45.0, 28.0)
        pluie = st.number_input("Pluviométrie (mm)", value=1100.0)

    # Calcul des stats pour les autres sections
    stats = df_yield[(df_yield['Area'] == pays) & (df_yield['Item'] == plante)]
    yield_moyen = stats['hg/ha_yield'].mean() if not stats.empty else 45000.0
    t_norm = stats['avg_temp'].mean() if not stats.empty else 24.0

    if st.button("🔍 GÉNÉRER LE RAPPORT D'ANALYSE"):
        conf = CONFIG_PLANTES[plante]
        age = len(st.session_state['hist_temp'])
        gdd_total = sum([max(0, t - conf['temp_base']) for t in st.session_state['hist_temp']])
        prog = min(100.0, (gdd_total / conf['seuil_mat']) * 100)
        stade = "Croissance" if gdd_total < conf['seuil_flo'] else "Maturation"
        
        input_data = pd.DataFrame([[pays, plante, t_site, pluie]], columns=['Area', 'Item', 'avg_temp', 'average_rain_fall_mm_per_year'])
        pred_yield = model_prod.predict(input_data)[0]
        perte_pct = ((yield_moyen - pred_yield) / yield_moyen) * 100

        st.code(f"""
╔════════════════════════════════════════════════════════════╗
║ 🛰️  RAPPORT D'ANALYSE : {plante.upper()} - {pays.upper()} ║
╠════════════════════════════════════════════════════════════╣
║ 🌱 ÉTAT : {stade} | 📅 ÂGE : {age} jours                 ║
║ 🌡️ TEMP SITE : {t_site:.1f}°C (Normal: {t_norm:.1f}°C)      ║
║ 📊 IMPACT IA : {perte_pct:+.1f}% | RENDEMENT : {pred_yield:.0f}  ║
╚════════════════════════════════════════════════════════════╝
""", language="text")

# --- 2. PROD LIVE ---
with tabs[1]:
    st.header("📡 Surveillance Production Live")
    t_live = t_site + np.random.uniform(-1.5, 1.5)
    st.metric("Capteur Parcelle", f"{t_live:.2f} °C", f"{t_live-t_site:+.2f}")
    st.line_chart(st.session_state['hist_temp'][-30:])
    
    if t_live > 33: st.error(f"DÉCISION : Stress thermique ! Activer l'aspersion.")
    else: st.success("DÉCISION : Paramètres optimaux.")

# --- 3. STOCK IA ---
with tabs[2]:
    st.header("🏗️ Diagnostic IA Stockage")
    cA, cB, cC = st.columns(3)
    s_cylindre = cA.number_input("Temp. Moteur Ventilation (S3)", value=82.0)
    s_valve = cB.number_input("Temp. Flux d'Air (S11)", value=52.0)
    s_ambiant = cC.number_input("Temp. Ambiante Silo (S18)", value=24.0)

    if st.button("🏗️ ANALYSER LE STOCK"):
        cluster = model_stock.predict(scaler_stock.transform([[s_cylindre, s_valve, s_ambiant]]))[0]
        msgs = ["🟢 STABLE", "🟡 VIGILANCE", "🔴 CRITIQUE (Fermentation)"]
        st.subheader(f"Statut IA : {msgs[cluster]}")
        

# --- 4. STOCK LIVE ---
with tabs[3]:
    st.header("📟 Monitoring Silos Live")
    ls3 = s_cylindre + np.random.uniform(-3, 3)
    st.metric("Sonde Flux S11 (Live)", f"{ls3:.1f}°C")
    st.area_chart(np.random.randn(20, 2))
    cl_live = model_stock.predict(scaler_stock.transform([[ls3, s_valve, s_ambiant]]))[0]
    if cl_live == 2: st.error("🚨 ALERTE LIVE : Air stagnant. Ouvrir les vannes S11.")

# --- 5. DISTRIBUTION & CLIENTS ---
with tabs[4]:
    st.header("🚛 Logistique & Portefeuille Clients")
    clients_db = {
        "BioFécule S.A.": "Transformateur - Luanda",
        "Manioc d'Or": "Grossiste - Montréal",
        "AgroExport Co.": "Exportateur - Paris",
        "DistriSud": "Détaillant - Lubango"
    }
    col_cl, col_form = st.columns([1, 2])
    with col_cl:
        st.subheader("Clients")
        for c, d in clients_db.items(): st.write(f"**{c}** : {d}")
    with col_form:
        with st.form("form_dist"):
            c_nom = st.selectbox("Client", list(clients_db.keys()))
            c_vol = st.number_input("Volume (T)", 1, 1000, 50)
            if st.form_submit_button("📦 Valider"):
                st.session_state['expeditions'].append({"ID": f"EXP-{len(st.session_state['expeditions'])+101}", "Client": c_nom, "Volume": f"{c_vol} T", "Statut": "En transit"})
    st.dataframe(pd.DataFrame(st.session_state['expeditions']), use_container_width=True)

# --- 6. PRÉVISIONS ---
with tabs[5]:
    st.header("📈 Prévisions de Rendement (2024-2028)")
    annees = [2024, 2025, 2026, 2027, 2028]
    data_prev = pd.DataFrame({"Année": annees, "Rendement Prédit": [yield_moyen * (1 + (np.random.uniform(-0.02, 0.05) * i)) for i in range(5)]})
    st.line_chart(data_prev.set_index("Année"))
    st.info(f"Tendance pour {plante} : Stabilité prévue avec une marge de variation de 3%.")

# --- 7. CONTACTS ---
with tabs[6]:
    st.header("📞 Réseau d'Experts")
    st.markdown(f"**Expert Principal :** Dr. Landry Takam | 📧 expert@agrimind.ai")