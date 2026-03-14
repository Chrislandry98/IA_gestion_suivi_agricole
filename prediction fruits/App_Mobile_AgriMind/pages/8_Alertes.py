import streamlit as st
import pandas as pd

st.set_page_config(page_title="AgriMind - Alertes", layout="wide")

# Initialisation session state
if 'alertes' not in st.session_state:
    st.session_state['alertes'] = []

st.header("🚨 Historique des alertes")

# Bouton pour réinitialiser les alertes
if st.button("🗑️ Réinitialiser les alertes"):
    st.session_state['alertes'] = []
    st.success("Toutes les alertes ont été réinitialisées !")

if st.session_state['alertes']:
    df_alertes = pd.DataFrame(st.session_state['alertes'])
    
    # Fonction pour colorer les cellules de la colonne 'Type'
    def color_type_cell(val):
        if "ROUGE" in val:
            return 'background-color:#FFB3B3'  # rouge clair
        elif "VIGILANCE" in val:
            return 'background-color:#FFF3CD'  # jaune clair
        else:
            return ''
    
    # Appliquer cellule par cellule uniquement sur 'Type'
    styled = df_alertes.style.applymap(color_type_cell, subset=['Type'])
    
    st.table(styled)
else:
    st.info("Aucune alerte détectée pour le moment.")