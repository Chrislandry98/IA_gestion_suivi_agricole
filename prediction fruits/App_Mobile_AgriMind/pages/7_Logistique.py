# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 15:12:20 2026

@author: Chris Landry
"""

import streamlit as st
import pandas as pd
from utils.ui import show_header

show_header()

st.title("🚛 Logistique")

dest = st.text_input("Destination")

vol = st.number_input("Volume")

if st.button("Créer expédition"):

    st.session_state.setdefault("expeditions",[]).append(
        {"Destination":dest,"Volume":vol}
    )

st.table(pd.DataFrame(st.session_state.get("expeditions",[])))