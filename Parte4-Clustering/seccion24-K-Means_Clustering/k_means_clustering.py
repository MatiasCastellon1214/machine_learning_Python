# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 19:58:21 2024

@author: Mirty
"""

# K - Means - Clustering

# Importamos las librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Importamos la base de datos con pandas

dataset = pd.read_csv("Mall_Customers.csv")

X = dataset.iloc[:, [3, 4]].values


# Método del codo para averiguar el número óptimo de clusters
from sklearn.cluster import KMeans

wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init='k-means++', max_iter = 300, n_init= 10, random_state= 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)


plt.plot(range(1, 11), wcss)
plt.title('Método del codo')
plt.xlabel('Número de clusters')
plt.ylabel('WCSS (k)')
plt.show()


# Aplicar el método de k-means para segmentar el dataset
kmeans = KMeans(n_clusters = 5, init='k-means++', max_iter = 300, n_init= 10, random_state= 0)
y_kmeans = kmeans.fit_predict(X)  


# Visualización de los clusters  
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Standard')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Objetivo')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cautos')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Conservadores')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Descuidados')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Baricentros')
plt.title('Cluster de clientes')
plt.xlabel('Ingresos anuales (en miles de $')
plt.ylabel('Puntuación de Gastos (1-100')
plt.legend()
plt.show()    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    