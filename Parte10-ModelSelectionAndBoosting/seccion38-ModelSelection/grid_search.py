# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 10:14:41 2025

@author: Mirty
"""

# Grid Search

# Importamos las librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Importamos la base de datos

dataset = pd.read_csv("Social_Network_Ads.csv")

X = dataset.iloc[:, [0,1]].values
y = dataset.iloc[:, 2].values

# Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)


# Escalado de  variables

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)


# Ajustar el clasificador con el Conjunto de Entrenamiento
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)


# Predecir el resultado con el conjunto de test
y_pred = classifier.predict(X_test)


# Elaborar una matriz de confusión (para evaluar las predicciones)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# Aplicar K-Flod cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

# Aplicar la mejora de Grid Search para optimizar el modelo y sus parámetros
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1, 10, 100, 1000],'kernel': ['linear']},
              {'C': [1, 10, 100, 1000],'kernel': ['rbf'], 'gamma': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]}
              ]

grid_search = GridSearchCV(estimator= classifier, 
                           param_grid= parameters,
                           scoring= 'accuracy',
                           cv= 10,
                           n_jobs= -1)

grid_search = grid_search.fit(X_train, y_train)

best_accuracy = grid_search.best_score_

best_parameters = grid_search.best_params_

# Representación gráfica de los resultados del algoritmo de Entrenamiento
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X1.min(), X1.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# Representación gráfica de los resultados del algoritmo de Testing
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X1.min(), X1.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Testing set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
