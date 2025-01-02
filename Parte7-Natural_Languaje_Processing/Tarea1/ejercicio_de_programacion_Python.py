# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 09:21:52 2024

@author: Mirty
"""

# Ejercicio de programación - Python


# Importamos las librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Importamos la abse de datos

dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting = 3)



# Limpieza de texto
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


corpus = []

for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)


# Crear el Bag of Words
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values


# Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)



# Entrenar diferentes modelos
# Importar librerías
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# Inicializar modelos
models = {
    "Logistic Regression": LogisticRegression(random_state=0),    
    "K Nearest Neighbors": KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2),
    "Support Vector Machine": SVC(kernel = 'linear', random_state = 0),
    "Naive Bayes": GaussianNB(),
    "Tree Decission Classification": DecisionTreeClassifier(criterion = "entropy", random_state = 0),
    "Random Forest": RandomForestClassifier(n_estimators = 10, criterion = "entropy", random_state=0)
}

# Entrenar los modelos
trained_models = {name: model.fit(X_train, y_train) for name, model in models.items()}


# Evaluar la eficacia de los modelos
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

for name, model in trained_models.items():
    print(f"Evaluating {name}")
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    # Métricas
    print("Confusion Matrix:")
    print(cm)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.2f}")
    print(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.2f}")
    print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("-" * 50)



# Visualización de los resultados - Tabla
results = []
for name, model in trained_models.items():
    y_pred = model.predict(X_test)
    results.append({
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 2),
        "Precision": round(precision_score(y_test, y_pred, average='weighted'), 2),
        "Recall": round(recall_score(y_test, y_pred, average='weighted'), 2),
        "F1 Score": round(f1_score(y_test, y_pred, average='weighted'), 2)
    })

df_results = pd.DataFrame(results)
print(df_results)


# Visualización de los resultados - Gráfico
df_results.plot(x="Model", kind="bar", figsize=(10, 6))
plt.title("Comparación de Modelos")
plt.ylabel("Puntuación")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()




# Inicializar modelos

models = {
    # CART
    "CART": DecisionTreeClassifier(criterion='gini', random_state=0),
    # Máxima Entropia
    "Maximun Entropy": LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=0),
    # C5.0
    "C5.0": DecisionTreeClassifier(criterion = 'entropy', max_depth=3, max_leaf_nodes=7), 
    "Logistic Regression": LogisticRegression(random_state=0),    
    "K Nearest Neighbors": KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2),
    "Support Vector Machine": SVC(kernel = 'linear', random_state = 0),
    "Naive Bayes": GaussianNB(),
    "Tree Decission Classification": DecisionTreeClassifier(criterion = "entropy", random_state = 0),
    "Random Forest": RandomForestClassifier(n_estimators = 10, criterion = "entropy", random_state=0)
}



# Entrenar los modelos
trained_models = {name: model.fit(X_train, y_train) for name, model in models.items()}


# Evaluar la eficacia de los modelos
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

for name, model in trained_models.items():
    print(f"Evaluating {name}")
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    # Métricas
    print("Confusion Matrix:")
    print(cm)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.2f}")
    print(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.2f}")
    print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("-" * 50)



# Visualización de los resultados - Tabla
results = []
for name, model in trained_models.items():
    y_pred = model.predict(X_test)
    results.append({
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 2),
        "Precision": round(precision_score(y_test, y_pred, average='weighted'), 2),
        "Recall": round(recall_score(y_test, y_pred, average='weighted'), 2),
        "F1 Score": round(f1_score(y_test, y_pred, average='weighted'), 2)
    })

df_results = pd.DataFrame(results)
print(df_results)


# Visualización de los resultados - Gráfico
df_results.plot(x="Model", kind="bar", figsize=(10, 6))
plt.title("Comparación de Modelos")
plt.ylabel("Puntuación")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()




















