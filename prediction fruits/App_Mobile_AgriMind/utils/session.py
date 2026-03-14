# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 14:53:47 2026

@author: Chris Landry
"""

import streamlit as st
import numpy as np
from datetime import datetime

def init_session():

    if "hist_temp" not in st.session_state:
        st.session_state.hist_temp = np.random.uniform(22,26,45).tolist()

    if "alertes" not in st.session_state:
        st.session_state.alertes = []

    if "messages_clients" not in st.session_state:
        st.session_state.messages_clients = [
            {
                "Client":"Admin",
                "Message":"Bienvenue sur AgriMind",
                "Date":datetime.now().strftime("%Y-%m-%d %H:%M")
            }
        ]