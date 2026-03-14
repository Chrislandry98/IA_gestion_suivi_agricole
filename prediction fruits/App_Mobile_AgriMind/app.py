# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 14:53:44 2026

@author: Chris Landry
"""


import streamlit as st


def load_css():

    with open("assets/style.css") as f:

        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

load_css()



import streamlit as st
from auth import login_page

st.set_page_config(
    page_title="AgriMind Global Expert",
    page_icon="🌱",
    layout="wide"
)

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    login_page()
else:
    st.switch_page("pages/1_Home.py")