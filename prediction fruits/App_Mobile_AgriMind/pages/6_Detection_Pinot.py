import streamlit as st
from PIL import Image
from ultralytics import YOLO
import os

st.set_page_config(page_title="🎯 Détection Pinot", layout="wide")

# --- CSS design agricole ---
st.markdown("""
<style>
h2 {color:white; text-align:center;}
body {background-color:#1b5e20;}
div.stButton > button {background-color:#2E7D32; color:white; border-radius:8px;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h2 style='text-align:center;'>🎯 Détection de Fruits (Pinot / YOLOv8)</h2>", unsafe_allow_html=True)

# --- Chargement modèle YOLOv8 ---
@st.cache_resource
def charger_yolo():
    # Mets ici le chemin absolu de ton fichier best.pt
    yolo_path = "C:/Users/Marco/Desktop/Hackaton/IA/prediction fruits/App_Mobile_AgriMind/models/best.pt"
    
    if os.path.exists(yolo_path):
        model = YOLO(yolo_path)
        st.success("Modèle YOLO Pinot chargé avec succès !")
        return model
    else:
        st.warning(f"⚠️ Modèle '{yolo_path}' non trouvé. La détection sera simulée.")
        return None

yolo_model = charger_yolo()

# --- Input utilisateur ---
col1, col2 = st.columns(2)
with col1:
    cam_input = st.camera_input("Scanner directement")
with col2:
    file_input = st.file_uploader("Ou charger une photo", type=["jpg","png","jpeg"])

input_image = cam_input if cam_input is not None else file_input

if input_image:
    img = Image.open(input_image).convert("RGB")
    st.image(img, caption="Image chargée", use_container_width=True)

    if yolo_model:
        with st.spinner("Analyse en cours..."):
            results = yolo_model(img)
            res_plot = results[0].plot()
            st.divider()

            # Affichage résultat centré
            cl, cm, cr = st.columns([1.5,2,1.5])
            with cm:
                st.image(res_plot, caption="Analyse terminée", use_container_width=True)
                nb_detect = len(results[0].boxes)
                st.metric("Nombre de fruits détectés", nb_detect)
                if nb_detect > 0:
                    st.success(f"✅ Détection réussie : {nb_detect} fruits identifiés")
                else:
                    st.warning("⚠️ Aucun fruit détecté. Vérifier l'image.")
    else:
        st.error("⚠️ Modèle YOLO 'best.pt' introuvable.")