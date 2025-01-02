# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 09:27:42 2024

@author: Mirty
"""
# Plantilla de Pre Procesado - Datos Categóricos


# Importamos las librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Importar el data set
dataset = pd.read_csv("Data.csv")

X = dataset.iloc[: , :-1].values
y = dataset.iloc[:, 3].values


# Codificar datos Categóricos
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

le = LabelEncoder() 
X[:, 0] = le.fit_transform(X[:, 0])

ct = ColumnTransformer( [('one_hot_encoder', OneHotEncoder(categories='auto'), [0])], 
                       remainder='passthrough')

X = np.array(ct.fit_transform(X), dtype=int)
y = le.fit_transform(y)
