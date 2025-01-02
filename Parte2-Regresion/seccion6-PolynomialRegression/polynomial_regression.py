# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 10:18:56 2024

@author: Mirty
"""

# Regresión Polinómica

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


'''
# En este caso, como el dataset tiene pocas filas, no hace falta dividirlo
# en conjunto de datos de entrenamiento y conjunto de testing
# Además, no se podría dividir la colummna Position porque son cargos jerárquicos
'''

## Vamos a comparar las regresiones lineal y polinómica

# Ajustar la regresión lineal con el dataset
from sklearn.linear_model import LinearRegression

lin_regression = LinearRegression()
lin_regression.fit(X, y)


# Ajuntar la regresión polinómica
from sklearn.preprocessing import PolynomialFeatures

# Regresión polinómica de grado 2

poly_regression = PolynomialFeatures(degree = 2)
X_poly = poly_regression.fit_transform(X)

lin_regression_2 = LinearRegression()
lin_regression_2.fit(X_poly, y)


# Visualización de los resultados del Modelo Lineal
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_regression.predict(X), color = 'blue')
plt.title('Modelo de Regresión Lineal')
plt.xlabel('Posición del empleado')
plt.ylabel('Suendo en $')
plt.show()

# Visualización de los resultados del Modelo Polinómico
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_regression_2.predict(X_poly), color = 'blue')
plt.title('Modelo de Regresión Polinómica')
plt.xlabel('Posición del empleado')
plt.ylabel('Suendo en $')
plt.show()

# Regresión polinómica de grado 3

poly_regression = PolynomialFeatures(degree = 3)
X_poly = poly_regression.fit_transform(X)

lin_regression_2 = LinearRegression()
lin_regression_2.fit(X_poly, y)

# Visualización de los resultados del Modelo Polinómico
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_regression_2.predict(X_poly), color = 'blue')
plt.title('Modelo de Regresión Polinómica')
plt.xlabel('Posición del empleado')
plt.ylabel('Suendo en $')
plt.show()


# Regresión polinómica de grado 4

poly_regression = PolynomialFeatures(degree = 4)
X_poly = poly_regression.fit_transform(X)

lin_regression_2 = LinearRegression()
lin_regression_2.fit(X_poly, y)


# Visualización de los resultados del Modelo Polinómico

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)

plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_regression_2.predict(poly_regression.fit_transform(X_grid)), color = 'blue')
plt.title('Modelo de Regresión Polinómica')
plt.xlabel('Posición del empleado')
plt.ylabel('Suendo en $')
plt.show()


# Decrepado: lin_reg.predict(6.5)
# Actualizado: lin_reg.predict([[6.5]])


# Predicción de nuestros modelos
lin_regression.predict([[6.5]])

lin_regression_2.predict(poly_regression.fit_transform([[6.5]]))

















