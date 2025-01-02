# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 09:30:40 2024

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

# Tratamiento de los NAs
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

