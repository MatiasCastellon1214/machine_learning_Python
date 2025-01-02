# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 15:03:30 2024

@author: Mirty
"""


# Random Selection

# Importamos librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Importamos el dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')


# Implementar Random Selection
import random

N = 10000
d = 10
ads_selected = []
total_rewards = 0
for n in range(0, N):
    ad = random.randrange(d)
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    total_rewards += reward
    

# Visualización de resultados - Histograma
plt.hist(ads_selected)
plt.title('Histograma de selección de anuncios')
plt.xlabel('Anuncio')
plt.ylabel('Número de veces que ha sido visualizado')
plt.show()
