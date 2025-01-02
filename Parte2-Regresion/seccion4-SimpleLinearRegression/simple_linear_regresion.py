# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 11:13:33 2024

@author: Mirty
"""

# Regresión Lineal Simple

# Importar Librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Importar el data set
dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values


# Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=1)

## Para una Regresiópn Lineal Simple no hace falta estalar los datos

# Crear modelo de Regreseión Lineal con el conjunto de Entrenamiento
from sklearn.linear_model import LinearRegression

regression = LinearRegression()
regression.fit(X_train, y_train)


# Predecir el conjunto de test
y_pred = regression.predict(X_test)


# Visualizar los resultados de entrenamiento
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regression.predict(X_train), color='blue')
plt.title("Sueldo vs Años de  Experiencia (Conjunto de Entrenamiento)")
plt.xlabel("Años de Experiencia")
plt.ylabel("Sueldo ($)")
plt.show()


# Visualizar los resultados de test
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regression.predict(X_train), color='blue')
plt.title("Sueldo vs Años de  Experiencia (Conjunto de Testing)")
plt.xlabel("Años de Experiencia")
plt.ylabel("Sueldo ($)")
plt.show()









