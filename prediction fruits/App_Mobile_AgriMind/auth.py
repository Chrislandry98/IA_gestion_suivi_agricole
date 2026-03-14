# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 14:53:46 2026

@author: Chris Landry
"""

import streamlit as st

def login_page():

    st.title("🔐 AgriMind Login")

    username = st.text_input("Utilisateur")
    password = st.text_input("Mot de passe", type="password")

    if st.button("Connexion"):

        if username == "admin" and password == "agrimind":

            st.session_state.authenticated = True
            st.success("Connexion réussie")

            st.switch_page("pages/1_Home.py")

        else:

            st.error("Identifiants incorrects")