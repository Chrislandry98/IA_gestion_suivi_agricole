# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 15:35:41 2026

@author: Chris Landry
"""

CONFIG_PLANTES = {
    'Cassava':{'temp_base':10,'seuil_floraison':1200,'seuil_maturite':2500,'nom':'Manioc','conseil':"Maintenir humidité stable.", 'sensibilite':1.5},
    'Maize':{'temp_base':8,'seuil_floraison':850,'seuil_maturite':1600,'nom':'Maïs','conseil':"Surveiller pollinisation.", 'sensibilite':1.2},
    'Potatoes':{'temp_base':5,'seuil_floraison':600,'seuil_maturite':1300,'nom':'Patate','conseil':"Eviter excès eau.", 'sensibilite':1.0},
    'Rice, paddy':{'temp_base':10,'seuil_floraison':1100,'seuil_maturite':2000,'nom':'Riz','conseil':"Niveau eau constant.", 'sensibilite':1.2},
    'Sorghum':{'temp_base':10,'seuil_floraison':900,'seuil_maturite':1800,'nom':'Sorgho','conseil':"Arrosage si stress thermique.", 'sensibilite':1.3},
    'Blueberry':{'temp_base':5,'seuil_floraison':300,'seuil_maturite':800,'nom':'Myrtille','conseil':"Eviter gel tardif.", 'sensibilite':1.4}
}