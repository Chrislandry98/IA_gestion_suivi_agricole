import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
import os

st.set_page_config(page_title="🫐 Vision Myrtilles", layout="wide")

# --- CSS design agricole ---
st.markdown("""
<style>
h1 {color:white; text-align:center;}
h2 {color:white; text-align:center;}
body {background-color:#1b5e20;}
div.stButton > button {background-color:#2E7D32; color:white; border-radius:8px;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h2 style='text-align:center;'>🫐 Diagnostic de Maturation des Myrtilles</h2>", unsafe_allow_html=True)

# --- Initialisation session_state ---
if 'hist_temp' not in st.session_state:
    import numpy as np
    st.session_state['hist_temp'] = np.random.uniform(22,26,45).tolist()
if 'messages_clients' not in st.session_state:
    st.session_state['messages_clients'] = []
if 'alertes' not in st.session_state:
    st.session_state['alertes'] = []

# --- Charger modèle Vision Myrtille ---
@st.cache_resource
def charger_ia_vision():
    model = models.mobilenet_v2(weights=None)
    num = model.classifier[1].in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.2),
        torch.nn.Linear(num, 128),
        torch.nn.ReLU(),
       torch.nn.Linear(128, 3),
       torch. nn.LogSoftmax(dim=1)
    )

    # Chemin absolu vers ton modèle
    model_path = "C:/Users/Marco/Desktop/Hackaton/IA/prediction fruits/App_Mobile_AgriMind/models/blueberry_recognition_model.pth"
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        st.success("Modèle Myrtille chargé avec succès !")
    else:
        st.warning(f"⚠️ Modèle '{model_path}' non trouvé. La détection sera simulée.")
    
    model.eval()
    return model

model_v = charger_ia_vision()

# --- Upload image et inference ---
up_file = st.file_uploader("📸 Prendre une photo de la grappe", type=["jpg","png","jpeg"])
if up_file:
    img = Image.open(up_file).convert('RGB')
    st.image(img, width=400)

    inference_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    input_tensor = inference_transforms(img).unsqueeze(0)

    with torch.no_grad():
        log_probs = model_v(input_tensor)
        probs = torch.exp(log_probs)
        conf, idx = torch.max(probs,1)

    classes = ['Immature','Mature','Semi-Mature']
    label = classes[idx.item()]

    st.subheader(f"Résultat : **{label}**")

    c1, c2, c3 = st.columns(3)
    c1.metric("Immature", f"{probs[0][0]*100:.1f}%")
    c2.metric("Mature", f"{probs[0][1]*100:.1f}%")
    c3.metric("Semi-Mature", f"{probs[0][2]*100:.1f}%")

    if label=="Mature":
        st.success("✅ Prêt pour la récolte.")
    else:
        st.warning("⏳ Attendre encore quelques jours.")