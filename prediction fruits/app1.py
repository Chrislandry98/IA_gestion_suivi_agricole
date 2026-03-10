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

# Initialisation de l'historique (pour le GDD et l'âge)
if 'hist_temp' not in st.session_state:
    # On simule 45 jours d'historique au départ
    st.session_state['hist_temp'] = np.random.uniform(22, 28, 45).tolist()

if 'expeditions' not in st.session_state:
    st.session_state['expeditions'] = []

# =================================================================
# 2. MOTEURS IA (SÉCURISÉS)
# =================================================================
@st.cache_resource
def charger_ia():
    base_path = os.path.dirname(__file__)
    prod_path = os.path.join(base_path, 'yield_df.csv')
    stock_path = os.path.join(base_path, 'archive/synthetic_industrial_data_with_status.csv')

    try:
        # PROD : On nettoie les noms pour éviter les "NaN"
        df_p = pd.read_csv(prod_path).dropna()
        df_p['Area'] = df_p['Area'].str.strip()
        df_p['Item'] = df_p['Item'].str.strip()
        
        model_p = Pipeline(steps=[
            ('preprocessor', ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), ['Area', 'Item'])], remainder='passthrough')),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        model_p.fit(df_p[['Area', 'Item', 'avg_temp', 'average_rain_fall_mm_per_year']], df_p['hg/ha_yield'])
        
        # STOCKAGE
        df_s = pd.read_csv(stock_path).dropna(subset=['sensor_3', 'sensor_11', 'sensor_18'])
        sc = StandardScaler()
        km = KMeans(n_clusters=3, random_state=42, n_init=10).fit(sc.fit_transform(df_s[['sensor_3', 'sensor_11', 'sensor_18']]))
        
        return df_p, model_p, km, sc
    except Exception as e:
        st.error(f"Erreur de chargement des fichiers CSV : {e}")
        st.stop()

df_yield, model_prod, model_stock, scaler_stock = charger_ia()

CONFIG_PLANTES = {
    'Cassava': {'temp_base': 10.0, 'seuil_flo': 1200, 'seuil_mat': 2500, 'nom': 'Manioc', 'conseil': "Stabilité hydrique requise."},
    'Maize': {'temp_base': 8.0, 'seuil_flo': 850, 'seuil_mat': 1600, 'nom': 'Maïs', 'conseil': "Surveiller la pollinisation."},
    'Rice, paddy': {'temp_base': 10.0, 'seuil_flo': 1100, 'seuil_mat': 2000, 'nom': 'Riz', 'conseil': "Niveau d'eau constant."}
}

# =================================================================
# 3. INTERFACE PRINCIPALE
# =================================================================
tabs = st.tabs(["🌾 PRODUCTION", "📡 PROD LIVE", "🏗️ STOCK IA", "📟 STOCK LIVE", "🚛 DISTRIBUTION", "📞 CONTACTS"])

# --- 1. PRODUCTION (RAPPORT EXPERT) ---
with tabs[0]:
    st.header("Analyse Agronomique Expert")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        pays = st.selectbox("Pays", sorted(df_yield['Area'].unique()))
        plante = st.selectbox("Culture", list(CONFIG_PLANTES.keys()))
        t_site = st.slider("Température Site (°C)", 10.0, 45.0, 28.0)
        pluie = st.number_input("Pluviométrie (mm)", value=1100.0)

    if st.button("🔍 GÉNÉRER LE RAPPORT D'ANALYSE"):
        conf = CONFIG_PLANTES[plante]
        # Âge et GDD
        age = len(st.session_state['hist_temp'])
        gdd_total = sum([max(0, t - conf['temp_base']) for t in st.session_state['hist_temp']])
        prog = min(100.0, (gdd_total / conf['seuil_mat']) * 100)
        stade = "Croissance" if gdd_total < conf['seuil_flo'] else "Maturation"
        
        # Calcul Climat (Sécurisé contre les NaN)
        stats = df_yield[(df_yield['Area'] == pays) & (df_yield['Item'] == plante)]
        t_norm = stats['avg_temp'].mean() if not stats.empty else 24.0
        yield_moyen = stats['hg/ha_yield'].mean() if not stats.empty else 1.0
        
        # IA Prediction
        input_data = pd.DataFrame([[pays, plante, t_site, pluie]], columns=['Area', 'Item', 'avg_temp', 'average_rain_fall_mm_per_year'])
        pred_yield = model_prod.predict(input_data)[0]
        ecart = t_site - t_norm
        perte_pct = ((yield_moyen - pred_yield) / yield_moyen) * 100

        st.code(f"""
╔════════════════════════════════════════════════════════════╗
║ 🛰️  RAPPORT D'ANALYSE AGRI-TECH : {plante.upper()} - {pays.upper()} ║
╠════════════════════════════════════════════════════════════╣
║ 📅 IDENTITÉ ET ÂGE                                         ║
║    • Âge de la culture : {age} jours
║    • Localisation      : {pays}
║    • Type de culture   : {conf['nom']} ({plante})
╠════════════════════════════════════════════════════════════╣
║ 🌱 ÉTAT PHYSIOLOGIQUE (MOTEUR GDD)                         ║
║    • Cumul Thermique : {gdd_total:.1f} GDD
║    • Stade Actuel    : {stade}
║    • Progression     : [{'█'*int(prog/5)}{'-'*int(20-prog/5)}] {prog:.1f}%
╠════════════════════════════════════════════════════════════╣
║ 🌡️  ANALYSE CLIMATIQUE                                     ║
║    • Température site : {t_site:.1f}°C (Normal: {t_norm:.1f}°C)
║    • Écart constaté  : {ecart:+.1f}°C
╠════════════════════════════════════════════════════════════╣
║ 📊 PRÉDICTION D'IMPACT (IA RANDOM FOREST)                  ║
║    • Impact rendement : {perte_pct:+.1f}%
║    • Statut           : {'⚠️ ALERTE BAISSE' if perte_pct > 5 else '✅ NORMAL'}
╠════════════════════════════════════════════════════════════╣
║ 📢 DÉCISION : {"🚨 ALERTE ROUGE" if ecart > 3 else "✅ ÉTAT STABLE"}
║ 💡 {conf['conseil']}
║ 💧 ACTION : {"IRRIGUER IMMEDIATEMENT" if ecart > 3 else "AUCUNE ACTION"}
╚════════════════════════════════════════════════════════════╝
""", language="text")

# --- 2. PROD LIVE (AUTONOME) ---
with tabs[1]:
    st.header("📡 Surveillance Production Live")
    t_live = t_site + np.random.uniform(-1.5, 1.5)
    st.subheader(f"Capteur Parcelle : {t_live:.2f} °C")
    st.line_chart(st.session_state['hist_temp'][-30:])
    
    if t_live > 33:
        st.error(f"DÉCISION LIVE : Stress thermique ! Activer l'aspersion sur {pays}.")
    else:
        st.success("DÉCISION LIVE : Paramètres optimaux. Poursuite du cycle.")

# --- 3. STOCK IA (EXPERT - NOMS RÉELS) ---
with tabs[2]:
    st.header("🏗️ Diagnostic IA Stockage")
    st.write("Saisissez les indicateurs réels des capteurs :")
    colA, colB, colC = st.columns(3)
    with colA: s_cylindre = st.number_input("Temp. Cylindre Compresseur (S3)", value=82.0)
    with colB: s_valve = st.number_input("Temp. Valves Aspiration (S11)", value=52.0)
    with colC: s_ambiant = st.number_input("Temp. Ambiante Silo (S18)", value=24.0)

    if st.button("🏗️ ANALYSER LE STOCK"):
        cluster = model_stock.predict(scaler_stock.transform([[s_cylindre, s_valve, s_ambiant]]))[0]
        if cluster == 0:
            st.success("🟢 STATUT : STABLE | Diagnostic : Conditions normales de conservation.")
        elif cluster == 1:
            st.warning("🟡 STATUT : VIGILANCE | Diagnostic : Hausse de l'humidité ou des frottements.")
        else:
            st.error("🔴 STATUT : CRITIQUE | Diagnostic : Risque élevé de fermentation ou moisissure.")

# --- 4. STOCK LIVE (AUTONOME) ---
with tabs[3]:
    st.header("📟 Monitoring Silos Live")
    ls3 = s_cylindre + np.random.uniform(-3, 3)
    st.metric("Capteur Cylindre (Live)", f"{ls3:.1f}°C")
    st.area_chart(np.random.randn(20, 2))
    
    cl_live = model_stock.predict(scaler_stock.transform([[ls3, s_valve, s_ambiant]]))[0]
    if cl_live == 2:
        st.error("🚨 RECOMMANDATION LIVE : Anomalie détectée. Ouvrir les vannes de ventilation S11.")
    else:
        st.success("✅ RECOMMANDATION LIVE : Stockage sécurisé.")

# --- 5. DISTRIBUTION & LIVRAISON ---
with tabs[4]:
    st.header("🚛 Gestion de la Logistique")
    with st.form("form_livraison"):
        col_d1, col_d2, col_d3 = st.columns(3)
        dest = col_d1.text_input("Destination")
        volume = col_d2.number_input("Volume (Tonnes)", min_value=1)
        prio = col_d3.selectbox("Priorité", ["Normale", "Haute", "Urgente"])
        submit = st.form_submit_button("Générer Ordre de Livraison")
        
        if submit:
            st.session_state['expeditions'].append({
                "ID": f"EXP-{len(st.session_state['expeditions'])+101}",
                "Destination": dest,
                "Volume": f"{volume} T",
                "Priorité": prio,
                "Statut": "Prêt pour départ"
            })
            st.success(f"Ordre de livraison vers {dest} enregistré.")

    st.subheader("📋 Historique des expéditions en cours")
    if st.session_state['expeditions']:
        st.table(pd.DataFrame(st.session_state['expeditions']))
    else:
        st.info("Aucune expédition programmée.")

# --- 6. CONTACTS & SUPPORT ---
with tabs[5]:
    st.header("📞 Réseau d'Experts Agri-Tech")
    c_left, c_right = st.columns(2)
    
    with c_left:
        st.markdown(f"""
        ### Expert Principal
        **Dr. Landry Takam** *Spécialiste en IA Opérationnelle & Agronomie* - 📧 **Email :** expert@agrimind.ai  
        - 📞 **Tel :** +244 9XX XXX XXX  
        - 📍 **Base :** Luanda / Montreal  
        """)
        st.button("🔄 Actualiser les flux Temps Réel", on_click=st.rerun)

    with c_right:
        st.markdown("""
        ### Support Technique IoT
        - **Maintenance Silos :** Équipe Delta-V
        - **Connectivité :** Réseau Satellite Starlink
        - **Urgence :** +244 800-AGRI-SAFE
        """)
        st.image("https://via.placeholder.com/400x150?text=Support+IOT+Actif", use_container_width=True)