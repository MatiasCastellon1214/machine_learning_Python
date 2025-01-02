# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 21:10:54 2024

@author: Mirty
"""

# Random Forest Regression


# Importamos las librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Importamos la abse de datos

dataset = pd.read_csv("Position_Salaries.csv")

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Ajustar el Random Forest con el dataset
# Crear aquí el modelo de regresión
from sklearn.ensemble import RandomForestRegressor

regression = RandomForestRegressor(n_estimators = 300, random_state= 0)
regression.fit(X, y)

# Predicción de nuestros modelos con Random Forest
y_pred = regression.predict([[6.5]]) 

# Visualización de los resultados del RFR

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regression.predict(X_grid), color = 'blue')
plt.title('Random Forest Regression')
plt.xlabel('Posición del empleado')
plt.ylabel('Salario')
plt.show()
