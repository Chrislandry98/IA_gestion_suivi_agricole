import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# =================================================================
# 1. BIBLIOTHÈQUE AGRONOMIQUE (Configuration spécifique par plante)
# =================================================================
# Ces valeurs définissent la biologie de chaque plante pour le calcul GDD
CONFIG_PLANTES = {
    'Cassava': {
        'temp_base': 10.0,
        'seuil_floraison': 1200,
        'seuil_maturite': 2500,
        'nom': 'Manioc',
        'conseil': "Maintenir une humidité stable pour le développement des tubercules."
    },
    'Maize': {
        'temp_base': 8.0,
        'seuil_floraison': 850,
        'seuil_maturite': 1600,
        'nom': 'Mais',
        'conseil': "Attention particulière à l'arrosage durant la phase de pollinisation."
    },
    'Potatoes': {
        'temp_base': 5.0,
        'seuil_floraison': 600,
        'seuil_maturite': 1300,
        'nom': 'Patate',
        'conseil': "Éviter les excès d'eau en fin de maturation pour prévenir le mildiou."
    },
    'Rice, paddy': {
        'temp_base': 10.0,
        'seuil_floraison': 1100,
        'seuil_maturite': 2000,
        'nom': 'riz',
        'conseil': "Niveau d'eau constant requis pour la phase de tallage."
    },
    'Sorghum': {
        'temp_base': 10.0,
        'seuil_floraison': 900,
        'seuil_maturite': 1800,
        'nom': 'Sorghum',
        'conseil': "Plante résistante, mais nécessite un arrosage d'appoint si l'écart > 5°C."
    }
}

# =================================================================
# 2. ENTRAÎNEMENT DU MODÈLE (Si non existant)
# =================================================================
def entrainer_si_besoin():
    if not os.path.exists('modele_maturation_agritech_final.pkl'):
        print("⚙️ Initialisation du modèle IA...")
        df = pd.read_csv('yield_df.csv')
        df['Area'] = df['Area'].str.strip()
        df['Item'] = df['Item'].str.strip()
        
        features = ['Area', 'Item', 'avg_temp', 'average_rain_fall_mm_per_year']
        target = 'hg/ha_yield'
        
        preprocessor = ColumnTransformer(
            transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), ['Area', 'Item'])],
            remainder='passthrough'
        )
        
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        
        model.fit(df[features], df[target])
        joblib.dump(model, 'modele_maturation_agritech_final.pkl')
        return df, model
    else:
        df = pd.read_csv('yield_df.csv')
        df['Area'] = df['Area'].str.strip()
        df['Item'] = df['Item'].str.strip()
        model = joblib.load('modele_maturation_agritech_final.pkl')
        return df, model

# =================================================================
# 3. MOTEUR DE GÉNÉRATION DU RAPPORT DÉTAILLÉ
# =================================================================

def generer_rapport_expert(pays, plante, historique_temp, pluie_actuelle):
    df, model = entrainer_si_besoin()
    
    # Configuration plante
    conf = CONFIG_PLANTES.get(plante, CONFIG_PLANTES['Cassava'])
    
    # Calcul GDD
    gdd_total = sum([max(0, t - conf['temp_base']) for t in historique_temp])
    age = len(historique_temp)
    
    # Stade physiologique
    if gdd_total < conf['seuil_floraison']:
        stade = "Croissance végétative"
        sensibilite = 1.5
        progression = (gdd_total / conf['seuil_floraison']) * 50 # Progression vers floraison
    elif gdd_total < conf['seuil_maturite']:
        stade = "Floraison / Fructification"
        sensibilite = 1.2
        progression = 50 + ((gdd_total - conf['seuil_floraison']) / (conf['s_mat'] - conf['seuil_floraison'])) * 50
    else:
        stade = "Maturation finale"
        sensibilite = 1.0
        progression = 100

    # Analyse IA
    temp_actuelle = historique_temp[-1]
    nom_p = [c for c in df['Area'].unique() if c.lower() == pays.lower().strip()][0]
    nom_i = [i for i in df['Item'].unique() if i.lower() == plante.lower().strip()][0]
    
    input_data = pd.DataFrame([[nom_p, nom_i, temp_actuelle, pluie_actuelle]], 
                             columns=['Area', 'Item', 'avg_temp', 'average_rain_fall_mm_per_year'])
    
    stats = df[(df['Area'] == nom_p) & (df['Item'] == nom_i)]
    temp_normale = stats['avg_temp'].mean()
    rendement_moyen = stats['hg/ha_yield'].mean()
    
    pred_yield = model.predict(input_data)[0]
    ecart = temp_actuelle - temp_normale
    perte_pct = ((rendement_moyen - pred_yield) / rendement_moyen) * 100

    # --- PRINT DU RAPPORT DÉTAILLÉ ---
    print("\n" + "╔" + "═"*58 + "╗")
    print(f"║ 🛰️  RAPPORT D'ANALYSE AGRI-TECH : {nom_i.upper()} - {nom_p.upper()} ║")
    print("╠" + "═"*58 + "╣")
    
    print(f"║ 📅 IDENTITÉ ET ÂGE                                      ║")
    print(f"║    • Âge de la culture : {age} jours                         ║")
    print(f"║    • Localisation      : {nom_p}                          ║")
    print(f"║    • Type de culture   : {conf['nom']} ({nom_i})            ║")
    
    print(f"╠" + "═"*58 + "╣")
    print(f"║ 🌱 ÉTAT PHYSIOLOGIQUE (MOTEUR GDD)                      ║")
    print(f"║    • Cumul Thermique : {gdd_total:.1f} GDD                       ║")
    print(f"║    • Stade Actuel    : {stade}             ║")
    print(f"║    • Progression     : [{'█'*int(progression/5)}{'-'*int(20-progression/5)}] {progression:.1f}%      ║")
    
    print(f"╠" + "═"*58 + "╣")
    print(f"║ 🌡️  ANALYSE CLIMATIQUE                                   ║")
    print(f"║    • Température actuelle sur site : {temp_actuelle:.1f}°C            ║")
    print(f"║    • Température normale (Historique) : {temp_normale:.1f}°C         ║")
    print(f"║    • Écart constaté  : {ecart:+.1f}°C                           ║")
    
    print(f"╠" + "═"*58 + "╣")
    print(f"║ 📊 PRÉDICTION D'IMPACT (IA RANDOM FOREST)               ║")
    print(f"║    • Impact sur le rendement final : {perte_pct:+.1f}%               ║")
    print(f"║    • Statut rendement : {'⚠️ ALERTE BAISSE' if perte_pct > 5 else '✅ NORMAL'}             ║")
    
    print(f"╠" + "═"*58 + "╣")
    
    # Logique de décision
    seuil_alerte = 4.0 / sensibilite
    if ecart > seuil_alerte:
        statut = "🚨 ALERTE ROUGE"
        texte_dec = "Stress thermique critique détecté."
        conseil = f"L'écart de {ecart:.1f}°C au stade '{stade}' est fatal."
        action = "IRRIGUER IMMÉDIATEMENT."
    elif ecart > (seuil_alerte/2):
        statut = "⚠️ VIGILANCE"
        texte_dec = "Risque de stress thermique modéré."
        conseil = "La plante commence à transpirer excessivement."
        action = "ARROSAGE PRÉVENTIF CONSEILLÉ."
    else:
        statut = "✅ ÉTAT STABLE"
        texte_dec = "Conditions conformes à l'historique."
        conseil = "Le métabolisme de la plante est optimal."
        action = "AUCUNE ACTION REQUISE."

    print(f"║ 📢 DÉCISION : {statut}                               ║")
    print(f"║ 📝 {texte_dec:<53} ║")
    print(f"║ 💡 {conseil:<53} ║")
    print(f"║ 💧 ACTION : {action:<43} ║")
    print("╚" + "═"*58 + "╝\n")

# =================================================================
# 4. SIMULATION
# =================================================================
if __name__ == "__main__":
    # Simulation de 45 jours se terminant par une canicule à 34°C
    historique = np.random.uniform(22, 26, 44).tolist()
    historique.append(34.0) 
    
    generer_rapport_expert("Angola", "Cassava", historique, 1100.0)