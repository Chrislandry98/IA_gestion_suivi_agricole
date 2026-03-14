# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 15:30:44 2026

@author: Chris Landry
"""

import streamlit as st

def show_header():

    st.markdown(
    """
    <div class="main-header">
    🌾 AGRIMIND GLOBAL EXPERT
    <br>
    Agriculture intelligente pilotée par IA
    </div>
    """,
    unsafe_allow_html=True
    )