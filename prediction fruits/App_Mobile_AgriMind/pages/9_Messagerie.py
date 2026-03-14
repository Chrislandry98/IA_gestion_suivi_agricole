# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 15:12:22 2026

@author: Chris Landry
"""

import streamlit as st
from datetime import datetime
import numpy as np

# --------------------------
# Initialisation des variables de session
# --------------------------
if 'messages_clients' not in st.session_state:
    st.session_state['messages_clients'] = [
        {"Client": "Admin",
         "Message": "Bienvenue sur AgriMind",
         "Date": datetime.now().strftime("%Y-%m-%d %H:%M")}
    ]

if 'alertes' not in st.session_state:
    st.session_state['alertes'] = []

if 'hist_temp' not in st.session_state:
    st.session_state['hist_temp'] = np.random.uniform(22,26,45).tolist()

if 'expeditions' not in st.session_state:
    st.session_state['expeditions'] = []

st.title("💬 Messagerie")



for m in st.session_state.messages_clients:

    st.write(
        f"{m['Client']} [{m['Date']}] : {m['Message']}"
    )

client = st.text_input("Nom")

msg = st.text_area("Message")

if st.button("Envoyer"):

    st.session_state.messages_clients.append({

        "Client":client,

        "Message":msg,

        "Date":datetime.now().strftime("%Y-%m-%d %H:%M")

    })

st.markdown(
f"""
    <div class="chat-bubble">
    <b>{m['Client']}</b><br>
    {m['Message']}
    </div>
""",
unsafe_allow_html=True
)