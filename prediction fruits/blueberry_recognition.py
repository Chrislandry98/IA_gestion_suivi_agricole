import streamlit as st
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# =================================================================
# 1. CONFIGURATION & ÉTAT DE LA SESSION
# =================================================================
st.set_page_config(page_title="AgriMind Global Expert", layout="wide")

CONFIG_PLANTES = {
    'Cassava': {'temp_base': 10.0, 'seuil_flo': 1200, 'seuil_mat': 2500, 'nom': 'Manioc'},
    'Maize': {'temp_base': 8.0, 'seuil_flo': 850, 'seuil_mat': 1600, 'nom': 'Maïs'},
    'Rice, paddy': {'temp_base': 10.0, 'seuil_flo': 1100, 'seuil_mat': 2000, 'nom': 'Riz'},
    'Blueberry': {'temp_base': 7.0, 'seuil_flo': 600, 'seuil_mat': 1100, 'nom': 'Myrtille', 'conseil': "Surveiller le pH du sol."}
}

if 'hist_temp' not in st.session_state:
    st.session_state['hist_temp'] = np.random.uniform(20, 30, 60).tolist()
if 'expeditions' not in st.session_state:
    st.session_state['expeditions'] = []

# =================================================================
# 2. CHARGEMENT DES MOTEURS IA
# =================================================================

@st.cache_resource
def charger_ia_vision():
    """Architecture EXACTE de ton script d'entraînement"""
    model = models.mobilenet_v2(weights=None)
    
    # REPRODUCTION DE TA TÊTE DE MODÈLE (Classifier)
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(num_ftrs, 128),  # Ta couche intermédiaire
        nn.ReLU(),
        nn.Linear(128, 3),         # Tes 3 classes
        nn.LogSoftmax(dim=1)       # Ton activation finale
    )
    
    # Chargement des poids
    path = 'blueberry_recognition_model.pth' # Assure-toi que le nom correspond
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    
    model.eval()
    return model

@st.cache_resource
def charger_ia_classique():
    base_path = os.path.dirname(__file__)
    prod_path = os.path.join(base_path, 'yield_df.csv')
    stock_path = os.path.join(base_path, 'archive/synthetic_industrial_data_with_status.csv')
    try:
        df_p = pd.read_csv(prod_path).dropna()
        model_p = Pipeline(steps=[
            ('preprocessor', ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), ['Area', 'Item'])], remainder='passthrough')),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        model_p.fit(df_p[['Area', 'Item', 'avg_temp', 'average_rain_fall_mm_per_year']], df_p['hg/ha_yield'])
        df_s = pd.read_csv(stock_path).dropna(subset=['sensor_3', 'sensor_11', 'sensor_18'])
        sc = StandardScaler()
        km = KMeans(n_clusters=3, random_state=42, n_init=10).fit(sc.fit_transform(df_s[['sensor_3', 'sensor_11', 'sensor_18']]))
        return df_p, model_p, km, sc
    except:
        return pd.DataFrame(), None, None, None

df_yield, model_prod, model_stock, scaler_stock = charger_ia_classique()

# =================================================================
# 3. INTERFACE STREAMLIT
# =================================================================
tabs = st.tabs(["🌾 PRODUCTION", "🫐 VISION MYRTILLES", "📡 LIVE", "🏗️ STOCK", "🚛 LOGISTIQUE"])

with tabs[0]:
    st.header("Analyse de Rendement")
    # ... (Code de sélection pays/plante identique)

with tabs[1]:
    st.header("🫐 Diagnostic de Maturation")
    up_file = st.file_uploader("Prendre une photo de la grappe", type=["jpg", "png", "jpeg"])
    
    if up_file:
        img = Image.open(up_file).convert('RGB')
        st.image(img, width=400)
        
        # TRANSFORMATION EXACTE (Celle de ton 'val_transforms')
        inference_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        with st.spinner("Analyse par l'IA..."):
            model_v = charger_ia_vision()
            input_tensor = inference_transforms(img).unsqueeze(0)
            
            with torch.no_grad():
                log_probs = model_v(input_tensor)
                probs = torch.exp(log_probs) # On repasse de LogSoftmax à Probabilité (0-1)
                conf, idx = torch.max(probs, 1)
            
            # L'ordre alphabétique automatique d'ImageFolder est :
            # 0: Immature, 1: Mature, 2: Semi-mature
            classes = ['Immature', 'Mature', 'Semi-Mature']
            label = classes[idx.item()]
            
            # Affichage des résultats
            st.subheader(f"Résultat : **{label}**")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Immature", f"{probs[0][0]*100:.1f}%")
            c2.metric("Mature", f"{probs[0][1]*100:.1f}%")
            c3.metric("Semi-Mature", f"{probs[0][2]*100:.1f}%")
            
            if label == "Mature":
                st.success("✅ Prêt pour la récolte.")
            else:
                st.warning("⏳ Attendre encore quelques jours.")

# ... (Le reste des onglets LIVE, STOCK, LOGISTIQUE reste identique)