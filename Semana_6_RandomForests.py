#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 14:11:15 2021

@author: lbarboza
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

#%%
# Generación de 4 nubes de puntos

from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=300, centers=4,
                  random_state=0, cluster_std=1.0)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='rainbow');

#%%
# Ajuste de un árbol de decisión a la nube de puntos y jerarquía

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text

ajuste_arbol = DecisionTreeClassifier().fit(X, y)

text_representation = export_text(ajuste_arbol)

#%%
# Visualización de la clasificación

def visualize_classifier(model, X, y, ax=None, cmap='rainbow'):
    ax = ax or plt.gca()
    # Plot the training points
    ax.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=cmap,
               clim=(y.min(), y.max()), zorder=3)
    ax.axis('tight')
    ax.axis('off')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # fit the estimator
    model.fit(X, y)
    xx, yy = np.meshgrid(np.linspace(*xlim, num=200),
                         np.linspace(*ylim, num=200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    # Create a color plot with the results
    n_classes = len(np.unique(y))
    contours = ax.contourf(xx, yy, Z, alpha=0.3,
                           levels=np.arange(n_classes + 1) - 0.5,
                           cmap=cmap, clim=(y.min(), y.max()),
                           zorder=1)
    ax.set(xlim=xlim, ylim=ylim)

visualize_classifier(DecisionTreeClassifier(), X, y)

#%%
# Para evitar el sobreajuste en los árboles de decisión, se puede
# utilizar bagging (por voto) para obtener un estimador con 
# menor varianza

from sklearn.ensemble import BaggingClassifier

arbol = DecisionTreeClassifier()
bagged = BaggingClassifier(arbol, n_estimators=100, max_samples=0.8,
                        random_state=1)
bagged.fit(X, y)
visualize_classifier(bagged, X, y)

#%%
# Random Forests permite hacer bagging de una forma optimizada, a 
# través de aleatoriedad en sus decisiones

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=0)
visualize_classifier(model, X, y);
 
#%%
# Ejemplo de clasificación de dígitos

from sklearn.datasets import load_digits

digits = load_digits()
print(digits.keys())

#%%
# Visualización de los dígitos
fig = plt.figure(figsize=(6, 6)) # tamaño de figura en pulgadas
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
# grafica los dígitos: cada imagen tiene 8X8 pixeles
for i in range(64):
    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
    ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')
    # etiqueta los dígitos con su valor objetivo
    ax.text(0, 7, str(digits.target[i]))

#%%
# Clasificación con Random Forests

from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(digits.data, digits.target,
                                                random_state=0)
model = RandomForestClassifier(n_estimators=1000)
model.fit(Xtrain, ytrain)
ypred = model.predict(Xtest)

#%%
# Métricas de la clasificación:
from sklearn import metrics
print(metrics.classification_report(ypred, ytest))
    
#%%
# Junto con la matriz de confusión
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(ytest, ypred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label');

