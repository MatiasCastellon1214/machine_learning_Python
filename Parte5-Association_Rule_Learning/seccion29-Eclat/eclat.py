# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 18:29:45 2024

@author: Mirty
"""

# Eclat


# Importamos librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importamos el dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)

# Acomodamos la tabla de otra forma más conveniente
transactions = []

for i in range(0, 7501): # Python no toma el último número. Por eso de 7501, porque queremos tomar 7500
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])


# SE ELIMINAN LOS PRODUCTOS 'NAN' DE LAS TRANSACCIONES
for i in range(0, len(transactions)):
    transactions[i] = [j for j in transactions[i] if str(j) != 'nan']
 

# Entrenar el algoritmo de Eclat
from apyori import apriori

rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2,
                min_lift = 3, min_linght = 2, max_lenght = 2)
# Para un mínimo de un 3 compras x día de un producto en una semana => min_support = 3*7/7500 = 0.0028


# Visualización de los resultados
results = list(rules)

results

## Putting the results well organised into a Pandas DataFrame
def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    return list(zip(lhs, rhs, supports))
resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Product 1', 'Product 2', 'Support'])

## Displaying the results non sorted
resultsinDataFrame

## Displaying the results sorted by descending lifts
resultsinDataFrame.nlargest(n = 10, columns = 'Support')








