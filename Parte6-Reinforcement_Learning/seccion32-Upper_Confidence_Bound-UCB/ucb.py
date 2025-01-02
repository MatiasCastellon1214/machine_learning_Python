# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 22:41:43 2024

@author: Mirty
"""

# Upper Confidence Bound (UCB)


# Importamos librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Importamos el dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

'''
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
'''


# Algoritmo de UCB
import math

N = 10000
d = 10

number_of_selection = [0] * d
sums_of_rewads = [0] * d
ads_selected = [0]
total_reward = 0

for n in range(0, N):
    max_upper_bound = 0
    ad = 0
    for i in range(0, d):
        if(number_of_selection[i] > 0):
            average_rewards = sums_of_rewads[i] / number_of_selection[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1) / number_of_selection[i])
            upper_bound = average_rewards + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    number_of_selection[ad] += 1 
    reward = dataset.values[n, ad]
    sums_of_rewads[ad] += reward
    total_reward += reward



# Histograma de resultados
plt.hist(ads_selected)
plt.title('Histograma de anuncios')
plt.xlabel('ID del Anuncio')
plt.ylabel('Frecuencia de visualización del anuncio')
plt.show()
















    
