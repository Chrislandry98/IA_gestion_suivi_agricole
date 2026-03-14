# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 15:39:01 2026

@author: Chris Landry
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from torchvision import models
from ultralytics import YOLO


@st.cache_resource
def charger_model_rf():

    df = pd.read_csv("C:/Users/Marco/Desktop/Hackaton/IA/prediction fruits/App_Mobile_AgriMind/models/data/yield_df.csv")

    try:
        model = joblib.load("models/modele_maturation_agritech_final.pkl")

    except:

        features = ['Area','Item','avg_temp','average_rain_fall_mm_per_year']
        target = 'hg/ha_yield'

        preprocessor = ColumnTransformer(
            [('cat',OneHotEncoder(handle_unknown='ignore'),['Area','Item'])],
            remainder='passthrough'
        )

        model = Pipeline([
            ('preprocessor',preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=150))
        ])

        model.fit(df[features],df[target])

        joblib.dump(model,"models/modele_maturation_agritech_final.pkl")

    return df, model


@st.cache_resource
def charger_stock():

    X = np.random.rand(200,3)
    model = KMeans(n_clusters=3,n_init=10)
    model.fit(X)

    return model


@st.cache_resource
def charger_ia_vision():

    model = models.mobilenet_v2(weights=None)

    num = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(num,128),
        nn.ReLU(),
        nn.Linear(128,3),
        nn.LogSoftmax(dim=1)
    )

    if os.path.exists("models/blueberry_recognition_model.pth"):

        model.load_state_dict(
            torch.load("models/blueberry_recognition_model.pth",
                       map_location="cpu")
        )

    model.eval()

    return model


@st.cache_resource
def charger_ia_pinot():

    if os.path.exists("models/best.pt"):
        return YOLO("models/best.pt")

    return None
