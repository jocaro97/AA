# -*- coding: utf-8 -*-

#############################
#####     LIBRERIAS     #####
#############################

import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas_summary import DataFrameSummary

import seaborn as sns

from IPython.core.display import display
from sklearn.model_selection import train_test_split
from sklearn import datasets, svm, preprocessing
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge
from sklearn.svm import LinearSVC
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings

# --------------------------------------------------------------------------------------
# Quitamos los warnings
warnings.filterwarnings('ignore')
# --------------------------------------------------------------------------------------
# Semilla
SEED = 100

# Leemos los ficheros
atributos = pd.read_csv("./datos/communitiesatrib.names")
print(atributos)
datos = pd.read_csv("./datos/communities.data", header = None, names = atributos["Atributos"])

datos_sum = DataFrameSummary(datos).summary()
print("Comprobamos que no hay valores perdidos en el dataset entrenamiento:")
display(datos_sum)

datos = datos.replace("?", np.nan)
print(len(datos))
datos_perdidos = datos.columns[datos.isnull().any()]
datos.dropna(axis = 'columns', inplace = True)
print(len(datos_perdidos))
print(len(datos))

datos = datos.drop(columns = ['communityname string'])


y = datos.iloc[:, -1]
X = datos.iloc[:, :-1]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.33, random_state = 42)

# Construimos los data frames
y_train_df = pd.DataFrame(data = y_train)
y_test_df = pd.DataFrame(data = y_test)

print(y_train_df)

# Comprobamos que las etiquetas estan dentro del intervalo que [0,1]
print("Todos los valores de las etiquetas de entrenamiento pertenecen al intervalo: [{},{}]".format(y_train_df.values.min(), y_train_df.values.max()))

print("Todos los valores de las etiquetas del conjunto de test pertenecen al intervalo: [{},{}]".format(y_test_df.values.min(), y_test_df.values.max()))

# Preprocesado
#preprocesado = [("PCA", PCA(n_components=0.95)),("escalado", StandardScaler())]
preprocesado = [("escalado", StandardScaler())]

preprocesador = Pipeline(preprocesado)

# Entrenamiento
# Regresion lineal

#lr = LinearRegression(fit_intercept = True)
lr = LinearRegression()
lr_pipe = Pipeline(steps=[('preprocesador', preprocesador),('clf',lr)])

sgd = (SGDRegressor(max_iter=1000, tol=1e-5))
params_sgd = {'clf__alpha':[1.0, 0.1, 0.01, 0.001, 0.0001]}

sgd_pipe = Pipeline(steps = [('preprocesador', preprocesador), ('clf', sgd)])

mejores_clasificadores = []

grid = GridSearchCV(sgd_pipe, params_sgd, scoring='r2', cv=5)

grid.fit(X_train,y_train)
mejores_clasificadores.append(grid.best_estimator_)
mejores_clasificadores.append(lr_pipe.fit(X_train,y_train))

ridge = (Ridge())
params_ridge = {'clf__alpha':[1.0, 0.1, 0.01, 0.001, 0.0001]}

ridge_pipe = Pipeline(steps = [('preprocesador', preprocesador), ('clf', ridge)])

grid = GridSearchCV(ridge_pipe, params_ridge, scoring='r2', cv=5)

grid.fit(X_train,y_train)
mejores_clasificadores.append(grid.best_estimator_)

score_sdg = mejores_clasificadores[0].score(X_train, y_train)
score_lr = mejores_clasificadores[1].score(X_train, y_train)
score_ridge = mejores_clasificadores[2].score(X_train, y_train)

print('score_sdg: {}'.format(score_sdg))
print('score_lr: {}'.format(score_lr))
print('score_ridge: {}'.format(score_ridge))

clasificador = mejores_clasificadores[0]
for i in range(len(mejores_clasificadores)):
	if(mejores_clasificadores[i].score(X_train,y_train) > clasificador.score(X_train,y_train)):
		clasificador = mejores_clasificadores[i]

clasificador.fit(X_train, y_train)

y_predict = clasificador.predict(X_test)


print("R^2: {:.4f}".format(r2_score(y_test, y_predict)))
print("Mean Squared Error: {}".format(mean_squared_error(y_test, y_predict)))
print("Mean Absolute Error: {}".format(mean_absolute_error(y_test, y_predict)))
