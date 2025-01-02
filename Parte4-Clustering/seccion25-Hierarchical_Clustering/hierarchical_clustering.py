# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 15:01:44 2024

@author: Mirty
"""

# Hierarchical Clustering


# Importamos las librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Importamos la base de datos con pandas

dataset = pd.read_csv("Mall_Customers.csv")

X = dataset.iloc[:, [3, 4]].values


# Utilizar el dendrograma para encontrar el número óptimo de clusters
import scipy.cluster.hierarchy as sch

dendrodram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrograma')
plt.xlabel('Clientes')
plt.ylabel('Distancia Euclídea')
plt.show()


# Ajustar el clustering jerárquico a nuestro conjunto de datos
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, metric = "euclidean", linkage = 'ward',)
y_hc = hc.fit_predict(X)


# Visualización de los clusters  
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Conservadores')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Standard')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Objetivo')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Descuidados')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cautos')
plt.title('Cluster de clientes')
plt.xlabel('Ingresos anuales (en miles de $')
plt.ylabel('Puntuación de Gastos (1-100')
plt.legend()
plt.show()    
    
    
    
















