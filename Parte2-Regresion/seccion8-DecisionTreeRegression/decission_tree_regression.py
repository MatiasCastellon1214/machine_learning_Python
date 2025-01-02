# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 19:48:59 2024

@author: Mirty
"""

# Decission Tree Regression

# Importamos las librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Importamos la abse de datos

dataset = pd.read_csv("Position_Salaries.csv")

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


# Ajustar la regresión con el dataset
from sklearn.tree import DecisionTreeRegressor

regression = DecisionTreeRegressor(random_state = 0)
regression.fit(X, y)

# Predicción de nuestro Modelo
y_pred = regression.predict([[6.5]])


# Visualización de los resultados del DTR
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = 'red')
plt.plot(X, regression.predict(X), color = 'blue')
plt.title('Modelo Polinómico')
plt.xlabel('Posición del empleado')
plt.ylabel('Salario')
plt.show()












