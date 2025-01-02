# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 11:49:53 2024

@author: Mirty
"""

# Importamos las librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Importamos el dataset
dataset = pd.read_csv("Position_Salaries.csv")

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


# Escalado de variables
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape(-1, 1))


# SVM

# Ajustar la regresión con el dataset
from sklearn.svm import SVR

regression = SVR(kernel="rbf")
regression.fit(X, y)

# Predicción del modelo con SVM
y_pred = sc_y.inverse_transform(regression.predict(sc_X.transform(np.array([[6.5], [4]]))).reshape(-1, 1))

# Visualización de los resultados del SVR
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)

plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regression.predict(X_grid), color = 'blue')
plt.title('Support Vector Machine')
plt.xlabel('Posición del empleado')
plt.ylabel('Salario')
plt.show()


##################################

'''
plt.scatter(dataset['Position'], dataset['Salary'])
plt.show()

# Regresiópn Lineal
from sklearn.linear_model import LinearRegression

lin_regression = LinearRegression()
lin_regression.fit(X, y)

plt.scatter(X, y, color='red')
plt.plot(X, lin_regression.predict(X), color='blue')
plt.title('Modelo Lineal')
plt.xlabel('Posición del empleado')
plt.ylabel('Salario')
plt.show()

# Regreasión Polinómica
from sklearn.preprocessing import PolynomialFeatures

# Grado 2
poly_regression = PolynomialFeatures(degree = 2)
X_poly = poly_regression.fit_transform(X)

lin_regression_2 = LinearRegression()
lin_regression_2.fit(X_poly, y)

plt.scatter(X, y, color = 'red')
plt.plot(X, lin_regression_2.predict(poly_regression.fit_transform(X)), color = 'blue')
plt.title('Modelo Polinómico')
plt.xlabel('Posición del empleado')
plt.ylabel('Salario')
plt.show()

# Grado 4
poly_regression = PolynomialFeatures(degree = 4)
X_poly = poly_regression.fit_transform(X)

lin_regression_2 = LinearRegression()
lin_regression_2.fit(X_poly, y)

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)

plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_regression_2.predict(poly_regression.fit_transform(X_grid)), color = 'blue')
plt.title('Modelo Polinómico')
plt.xlabel('Posición del empleado')
plt.ylabel('Salario')
plt.show()

'''



















