import streamlit as st
import pandas as pd
import numpy as np
import joblib, os, torch, torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from datetime import datetime

st.set_page_config(page_title="AgriMind Global Expert", layout="wide")

# --------------------------
# Configuration des plantes
# --------------------------
CONFIG_PLANTES = {
    'Cassava':{'temp_base':10,'seuil_floraison':1200,'seuil_maturite':2500,'nom':'Manioc','conseil':"Maintenir humidité stable."},
    'Maize':{'temp_base':8,'seuil_floraison':850,'seuil_maturite':1600,'nom':'Maïs','conseil':"Surveiller pollinisation."},
    'Potatoes':{'temp_base':5,'seuil_floraison':600,'seuil_maturite':1300,'nom':'Patate','conseil':"Eviter excès eau."},
    'Rice, paddy':{'temp_base':10,'seuil_floraison':1100,'seuil_maturite':2000,'nom':'Riz','conseil':"Niveau eau constant."},
    'Sorghum':{'temp_base':10,'seuil_floraison':900,'seuil_maturite':1800,'nom':'Sorgho','conseil':"Arrosage si stress thermique."}
}

# --------------------------
# Session state
# --------------------------
if 'hist_temp' not in st.session_state:
    st.session_state['hist_temp'] = np.random.uniform(22,26,45).tolist()
if 'alertes' not in st.session_state:
    st.session_state['alertes'] = []

# --------------------------
# Charger ou entraîner Random Forest rendement
# --------------------------
@st.cache_resource
def charger_model_rf():
    df = pd.read_csv("yield_df.csv")
    df['Area'] = df['Area'].str.strip()
    df['Item'] = df['Item'].str.strip()
    try:
        model = joblib.load("modele_maturation_agritech_final.pkl")
    except:
        features = ['Area','Item','avg_temp','average_rain_fall_mm_per_year']
        target = 'hg/ha_yield'
        preprocessor = ColumnTransformer([('cat',OneHotEncoder(handle_unknown='ignore'),['Area','Item'])], remainder='passthrough')
        model = Pipeline([('preprocessor',preprocessor), ('regressor', RandomForestRegressor(n_estimators=150, random_state=42))])
        model.fit(df[features],df[target])
        joblib.dump(model,"modele_maturation_agritech_final.pkl")
    return df, model

df_yield, model_yield = charger_model_rf()

# --------------------------
# Charger Stock IA
# --------------------------
@st.cache_resource
def charger_stock():
    X = np.random.rand(200,3)
    model = KMeans(n_clusters=3,n_init=10)
    model.fit(X)
    return model

model_stockage = charger_stock()

# --------------------------
# Charger Vision Myrtille
# --------------------------
@st.cache_resource
def charger_ia_vision():
    model = models.mobilenet_v2(weights=None)
    num = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(num,128), nn.ReLU(), nn.Linear(128,3), nn.LogSoftmax(dim=1))
    if os.path.exists("blueberry_recognition_model.pth"):
        model.load_state_dict(torch.load("blueberry_recognition_model.pth",map_location="cpu"))
    model.eval()
    return model

# --------------------------
# Fonction rapport expert
# --------------------------
def generer_rapport(pays,plante,temp_actuelle,pluie_actuelle):
    conf = CONFIG_PLANTES[plante]
    st.session_state['hist_temp'].append(temp_actuelle)
    historique_temp = st.session_state['hist_temp']
    gdd_total = sum(max(0,t-conf['temp_base']) for t in historique_temp)
    age = len(historique_temp)

    # Stade physiologique + progression + sensibilité
    if gdd_total < conf['seuil_floraison']:
        stade = "Croissance végétative"; progression=(gdd_total/conf['seuil_floraison'])*50; sensibilite=1.5
    elif gdd_total < conf['seuil_maturite']:
        stade = "Floraison / Fructification"; progression=50+((gdd_total-conf['seuil_floraison'])/(conf['seuil_maturite']-conf['seuil_floraison']))*50; sensibilite=1.2
    else:
        stade = "Maturation finale"; progression=100; sensibilite=1.0

    # Prévision rendement
    stats = df_yield[(df_yield['Area']==pays)&(df_yield['Item']==plante)]
    temp_normale = stats['avg_temp'].mean() if not stats.empty else temp_actuelle
    rendement_moyen = stats['hg/ha_yield'].mean() if not stats.empty else 0
    input_data = pd.DataFrame([[pays,plante,temp_actuelle,pluie_actuelle]],columns=['Area','Item','avg_temp','average_rain_fall_mm_per_year'])
    pred_yield = model_yield.predict(input_data)[0]
    perte_pct = ((rendement_moyen - pred_yield)/max(rendement_moyen,1))*100
    perte_pct = np.clip(perte_pct,-40,40)

    # Décision IA
    seuil_alerte = 4.0 / sensibilite
    if temp_actuelle - temp_normale > seuil_alerte:
        statut = "🚨 ALERTE ROUGE"; texte_dec="Stress thermique critique"; conseil="IRRIGUER IMMÉDIATEMENT"
    elif temp_actuelle - temp_normale > seuil_alerte/2:
        statut = "⚠️ VIGILANCE"; texte_dec="Risque modéré"; conseil="ARROSAGE PRÉVENTIF"
    else:
        statut = "✅ ÉTAT STABLE"; texte_dec="Conditions normales"; conseil="Aucune action requise"

    # Ajouter alerte session
    st.session_state['alertes'].append({"Date":datetime.now().strftime("%Y-%m-%d %H:%M"), "Pays":pays, "Plante":plante, "Temp":temp_actuelle, "Statut":statut, "Action":conseil})

    # Rapport texte
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

# --------------------------
# Interface principale
# --------------------------
tabs = st.tabs(["🌾 PRODUCTION","🫐 VISION MYRTILLES","📡 LIVE / 32 jours","🏗️ STOCK IA","🚛 LOGISTIQUE","🚨 ALERTES"])

# --------------------------
# PRODUCTION
# --------------------------
with tabs[0]:
    st.header("Analyse de Rendement")
    pays = st.selectbox("Pays", df_yield['Area'].unique())
    plante = st.selectbox("Culture", list(CONFIG_PLANTES.keys()))
    pluie = st.number_input("Pluie annuelle", value=1100)
    temp_actuelle = st.number_input("Température actuelle (°C)", value=st.session_state['hist_temp'][-1])

    if st.button("Générer rapport"):
        rapport = generer_rapport(pays,plante,temp_actuelle,pluie)
        st.code(rapport)

# --------------------------
# VISION MYRTILLES
# --------------------------
with tabs[1]:
    st.header("🫐 Diagnostic de Maturation")
    up_file = st.file_uploader("Prendre une photo de la grappe", type=["jpg","png","jpeg"])
    if up_file:
        img = Image.open(up_file).convert('RGB')
        st.image(img,width=400)
        inference_transforms = transforms.Compose([
            transforms.Resize((224,224)), transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        model_v = charger_ia_vision()
        input_tensor = inference_transforms(img).unsqueeze(0)
        with torch.no_grad():
            log_probs = model_v(input_tensor)
            probs = torch.exp(log_probs)
            conf, idx = torch.max(probs,1)
        classes = ['Immature','Mature','Semi-Mature']
        label = classes[idx.item()]
        st.subheader(f"Résultat : **{label}**")
        c1,c2,c3 = st.columns(3)
        c1.metric("Immature", f"{probs[0][0]*100:.1f}%")
        c2.metric("Mature", f"{probs[0][1]*100:.1f}%")
        c3.metric("Semi-Mature", f"{probs[0][2]*100:.1f}%")
        if label=="Mature": st.success("✅ Prêt pour la récolte.")
        else: st.warning("⏳ Attendre encore quelques jours.")

# --------------------------
# LIVE / 32 jours avec prévision
# --------------------------
with tabs[2]:
    st.header("Prévision 32 jours et recommandations")
    # Simulation API météo réelle ou intégration ultérieure
    future_temp = np.random.uniform(22,35,32)
    future_pluie = np.random.uniform(0,20,32)
    st.line_chart(future_temp)
    st.write("Température prévue (32j)")
    st.write(future_temp)
    st.write("Pluie prévue (32j)")
    st.write(future_pluie)
    # Calcul GDD prévisionnel et recommandations automatiques
    gdd_pred = sum(max(0,t-CONFIG_PLANTES[plante]['temp_base']) for t in st.session_state['hist_temp']+future_temp.tolist())
    if gdd_pred > CONFIG_PLANTES[plante]['seuil_maturite']:
        st.success("✅ La culture sera mature dans les prochains jours. Récolte possible.")
    elif gdd_pred > CONFIG_PLANTES[plante]['seuil_floraison']:
        st.info("⚠️ Phase de floraison / fructification. Surveiller arrosage.")
    else:
        st.warning("🌱 Croissance végétative. Prévoir irrigation si nécessaire.")

# --------------------------
# STOCK IA
# --------------------------
with tabs[3]:
    st.header("Analyse Stockage Silo")
    temp = st.slider("Température",0,60,25)
    hum = st.slider("Humidité",0,100,50)
    co2 = st.slider("CO2",0,2000,400)
    if st.button("Analyser silo"):
        data = np.array([[temp/60,hum/100,co2/2000]])
        cluster = model_stockage.predict(data)[0]
        if cluster==0:
            st.success("Stockage stable. Conseils : Maintenir conditions.")
        elif cluster==1:
            st.warning("Surveillance recommandée. Conseils : Ajuster ventilation / humidité.")
        else:
            st.error("Risque fermentation ! Conseils : Vérifier température et CO2.")

# --------------------------
# LOGISTIQUE
# --------------------------
with tabs[4]:
    st.header("Gestion Logistique")
    dest = st.text_input("Destination")
    vol = st.number_input("Volume",1)
    if st.button("Créer expédition"):
        st.session_state.setdefault('expeditions',[]).append({"Destination":dest,"Volume":vol})
    st.table(pd.DataFrame(st.session_state.get('expeditions',[])))

# --------------------------
# ALERTES
# --------------------------
with tabs[5]:
    st.header("Historique des alertes")
    if st.session_state['alertes']:
        st.table(pd.DataFrame(st.session_state['alertes']))
    else:
        st.info("Aucune alerte pour l'instant.")