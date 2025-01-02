# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 14:51:50 2024

@author: Mirty
"""

# Plantilla de Regresión


# Cargamos las librerias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Importamos el dataset

dataset = pd.read_csv("Position_Salaries.csv")
print(dataset.head())


plt.scatter(dataset['Level'], dataset['Salary'])
plt.show()

# En X es importante considerar 1:2 1n lugar de colocar 1 en la selección de columnas:
    # 1:2 = matriz de dos dimensiones (matriz de características)
    # 1 = vector de una dimensión
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Dividir el dataset en conjunto de entrenamiento y conjunto de testing
'''
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
'''


# Escalado de  variables
'''
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
'''

# Ajustar la regresión con el dataset
# Crear aquí nuestro moodel de regresión




# Predicción de nuestros modelos
y_pred = regression.predict(6.5) # Hará crear el objeto regression (lineal, polinómico, logístico, etc)




# Visualización de los resultados del Modelo
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)

plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regression.predict(X_grid), color = 'blue')
plt.title('Modelo de Regresión')
plt.xlabel('Posición del empleado')
plt.ylabel('Suendo en $')
plt.show()






















