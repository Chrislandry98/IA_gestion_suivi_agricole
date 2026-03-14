# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 14:53:47 2026

@author: Chris Landry
"""

import streamlit as st
from utils.session import init_session

init_session()

st.title("🌍 AgriMind Global Expert")

st.markdown(
"""
<div class="main-header">
🌾 AGRIMIND - Intelligent Agriculture Analytics & Crop Monitoring
<br>
Plateforme IA pour Agriculture Intelligente
</div>
""",
unsafe_allow_html=True
)

col1,col2,col3,col4 = st.columns(4)

with col1:
    st.markdown(
    """
    <div class="dashboard-card">
    🌾
    <h3>Production</h3>
    Analyse rendement
    </div>
    """,
    unsafe_allow_html=True
    )

with col2:
    st.markdown(
    """
    <div class="dashboard-card">
    📡
    <h3>Monitoring</h3>
    Capteurs IoT
    </div>
    """,
    unsafe_allow_html=True
    )

with col3:
    st.markdown(
    """
    <div class="dashboard-card">
    🏗
    <h3>Stockage</h3>
    Analyse silo
    </div>
    """,
    unsafe_allow_html=True
    )

with col4:
    st.markdown(
    """
    <div class="dashboard-card">
    🚛
    <h3>Logistique</h3>
    Distribution
    </div>
    """,
    unsafe_allow_html=True
    )

st.image(
"https://images.unsplash.com/photo-1500382017468-9049fed747ef",
use_container_width=True,
caption="Agriculture intelligente"
)

c1,c2,c3 = st.columns(3)

c1.metric("Température moyenne","26°C","+1.2°C")

c2.metric("Production estimée","12.4 T","+4%")

c3.metric("Alertes","2","-1")




