#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 11:10:12 2021

@author: maikol
"""
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np


rng = np.random.RandomState(1)
x = 10 * rng.rand(50)
y = 2 * x - 5 + rng.randn(50)
plt.scatter(x, y);


from sklearn.linear_model import LinearRegression
model = LinearRegression(fit_intercept=True)
model.fit(x[:, np.newaxis], y)
xfit = np.linspace(0, 10, 1000)
yfit = model.predict(xfit[:, np.newaxis])

plt.figure();
plt.scatter(x, y)
plt.plot(xfit, yfit);

print("Model slope:", model.coef_[0])
print("Model intercept:", model.intercept_)


"""
Si se deseara agregar coeficientes polinomiales, basta con importar la libreria PolynomialFeatures

La variable x contiene un ejemplo de los puntos donde se evaluaria ese polinomio de orden 3. 
"""

from sklearn.preprocessing import PolynomialFeatures
x = np.array([2, 3, 4])
poly = PolynomialFeatures(3, include_bias=False)
poly.fit_transform(x[:, None])

print(x)
print(poly)
print(poly.fit_transform(x[:, None]))

"""
Para hacer más sencillo la construcción de modelos, se puede usar la función 
make_pipeline la cual agrega layers de complejidad al modelo final
"""

from sklearn.pipeline import make_pipeline
poly_model = make_pipeline(PolynomialFeatures(7),
LinearRegression())


rng = np.random.RandomState(1)
x = 10 * rng.rand(50)
y = np.sin(x) + 0.1 * rng.randn(50)
poly_model.fit(x[:, np.newaxis], y)
yfit = poly_model.predict(xfit[:, np.newaxis])

plt.figure()
plt.scatter(x,y)
plt.plot(xfit,yfit)

"""
En este ejemplo vamos a construir una base gaussiana en lugar de una polinomial.

Una explicación del teórica de lo que hace el código se puede encontrar acá 

https://www.cs.princeton.edu/courses/archive/fall18/cos324/files/basis-functions.pdf

Resumiendo, lo que hace es que genera multiples columnas  a la matriz de diseño, 
donde cada columna representa un función gaussia centrada en cada uno de los puntos 
de la muestra.

"""


from sklearn.base import BaseEstimator, TransformerMixin

class GaussianFeatures(BaseEstimator, TransformerMixin):
    """Uniformly spaced Gaussian features for one-dimensional input"""
    def __init__(self, N, width_factor=2.0):
        self.N = N
        self.width_factor = width_factor
        
    @staticmethod     
    def _gauss_basis(x, y, width, axis=None):
        arg = (x - y) / width
        return np.exp(-0.5 * np.sum(arg ** 2, axis))
    
    def fit(self, X, y=None):
        # create N centers spread along the data range
        self.centers_ = np.linspace(X.min(), X.max(), self.N)
        self.width_ = self.width_factor * (self.centers_[1] - self.centers_[0])
        return self
        
    def transform(self, X):
        return self._gauss_basis(X[:, :, np.newaxis], self.centers_,
                                                 self.width_, axis=1)
gauss_model = make_pipeline(GaussianFeatures(20),
                            LinearRegression())
gauss_model.fit(x[:, np.newaxis], y)
yfit = gauss_model.predict(xfit[:, np.newaxis])

plt.figure()
plt.scatter(x, y)
plt.plot(xfit, yfit)
plt.xlim(0, 10);


"""
El ejemplo anterior solo pretende ilustrar el que pasaría si se tienen múltiples
columnas en nuestra base, pero algunas tienen información redudante
"""

model = make_pipeline(GaussianFeatures(30),
                      LinearRegression())
model.fit(x[:, np.newaxis], y)

plt.figure()
plt.scatter(x, y)
plt.plot(xfit, model.predict(xfit[:, np.newaxis]))
plt.xlim(0, 10)
plt.ylim(-1.5, 1.5);


"""
Esta función identifica el problema del caso anterior
"""

def basis_plot(model, title=None):
    plt.figure()
    fig, ax = plt.subplots(2, sharex=True)
    model.fit(x[:, np.newaxis], y)
    ax[0].scatter(x, y)
    ax[0].plot(xfit, model.predict(xfit[:, np.newaxis]))
    ax[0].set(xlabel='x', ylabel='y', ylim=(-1.5, 1.5))
    
    if title:
        ax[0].set_title(title)
        
    ax[1].plot(model.steps[0][1].centers_,
               model.steps[1][1].coef_)
    ax[1].set(xlabel='basis location',
              ylabel='coefficient',
              xlim=(0, 10))
    
model = make_pipeline(GaussianFeatures(30), LinearRegression())
basis_plot(model)


"""
Aplicando una regularización de Ridge a los datos, es bastante sencillo usando make_pipeline
"""

from sklearn.linear_model import Ridge
model = make_pipeline(GaussianFeatures(30), Ridge(alpha=0.1))
basis_plot(model, title='Ridge Regression')

print("Coeficientes para Ridge:")
print(model.steps[1][1].coef_)

"""
También se puede aplicar una regularización Lasso. Compare los resultados. S
"""

from sklearn.linear_model import Lasso
model = make_pipeline(GaussianFeatures(30), Lasso(alpha=0.001))
basis_plot(model, title='Lasso Regression')

print("Coeficientes para Lasso:")
print(model.steps[1][1].coef_)