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
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier, Perceptron, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

import warnings

# --------------------------------------------------------------------------------------
# Quitamos los warnings
warnings.filterwarnings('ignore')
# --------------------------------------------------------------------------------------
# Semilla
SEED = 100

# Funcion para leer los datos
def readData(filename):
	# Leemos los ficheros
	data = pd.read_csv(filename, header = None)
	y = data.iloc[:, -1]
	x = data.iloc[:, :-1]

	return x, y

# Lectura de los datos de entrenamiento
X, y = readData("./datos/optdigits.tra")
X_test, y_test = readData("./datos/optdigits.tes")

# Construimos los data frames
y_df = pd.DataFrame(data = y)
y_test_df = pd.DataFrame(data = y_test)

# Cambiamos el nombre de las columnas para mostrarlo mejor
X = X.add_prefix("Característica ")
X_test = X_test.add_prefix("Característica ")
y_df.rename(columns={64:"Dígito"}, inplace = True)
y_test_df.rename(columns={64:"Dígito"}, inplace = True)

# Comprobamos que no hay ningún valor perdido y también que
# ningún valor se salga del intervalo [0,16]
train_df = pd.DataFrame(data = np.c_[X,y])
train_sum = DataFrameSummary(train_df).summary()
print("Comprobamos que no hay valores perdidos en el dataset entrenamiento:")
display(train_sum)

test_df = pd.DataFrame(data = np.c_[X_test,y_test])
test_sum = DataFrameSummary(test_df).summary()
print("Comprobamos que no hay valores perdidos en el dataset de test:")
display(test_sum)

print("Hay {} datos de enternamiento".format(X.shape[0]))
print("Hay {} datos de test".format(X_test.shape[0]))

porc_train = 100*X.shape[0]/(X.shape[0]+ X_test.shape[0])
porc_test = 100*X_test.shape[0]/(X.shape[0]+ X_test.shape[0])
print("Hay {}% de datos de entrenamiento y un {}% de datos de test.".format(porc_train,porc_test))

print(y_df)
# Comprobamos que las clases estan balanceadas
print("Dataset de entrenamiento")
for i in range (len(y_df['Dígito'].unique())):
	print("\t {} instancias del dígito {}".format(y_df['Dígito'].value_counts()[i],i))
print("\n")
print("Dataset de test")
for i in range (len(y_df['Dígito'].unique())):
	print("\t {} instancias del dígito {}".format(y_test_df['Dígito'].value_counts()[i],i))

clases = np.unique(train_df.values[:,-1])
numero_elementos = []
for i in clases:
	numero_elementos.append(y_df['Dígito'].value_counts()[i])

df_plot = pd.DataFrame(columns= ["Dígitos", "Número de ejemplos"], data =[[c,n] for c, n in zip(clases,numero_elementos)])
sns.barplot(x="Dígitos", y ="Número de ejemplos", data = df_plot)
plt.title("Número de ejemplos de cada clase")
plt.show()

# Comprobamos que las etiquetas estan dentro del intervalo que [0,9]
print("Todos los valores de las etiquetas de entrenamiento pertenecen al intervalo: [{},{}]".format(y_df.values.min(), y_df.values.max()))

print("Todos los valores de las etiquetas del conjunto de test pertenecen al intervalo: [{},{}]".format(y_test_df.values.min(), y_test_df.values.max()))

# Preprocesado
preprocesado = [("varianza", VarianceThreshold(threshold=0.0)),
                ("escalado", StandardScaler()),
                ("PCA", PCA(n_components=0.95))]

preprocesador = Pipeline(preprocesado)

def mostrar_correlaciones(datos):
	f, ax = plt.subplots(figsize=(10, 8))
	corr = datos.corr()
	sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), square=True, ax=ax)
	f.suptitle('Matriz Correlaciones')
	plt.show()

mostrar_correlaciones(X)

def muestra_correlaciones_procesados(datos):
	f, ax = plt.subplots(figsize=(10, 8))
	corr = np.corrcoef(datos.T)
	sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),square=True, ax=ax)
	f.suptitle('Matriz Correlaciones')
	plt.show()

datos_preprocesados = preprocesador.fit_transform(X)
muestra_correlaciones_procesados(datos_preprocesados)

# Entrenamiento

# LinearSVC
svc = (LinearSVC(penalty='12', multi_class='crammer_singer', loss='hinge'))

svc_pipe = Pipeline(steps=[('preprocesador', preprocesador),('clf', svc)])

params_svc = {'clf__C': [2.0, 1.0, 0.1, 0.01, 0.001]}

# LogisticRegression
log = (LogisticRegression(penalty='l2', # Regularización Ridge (L2)
						  multi_class='multinomial', # Indicamos que la regresión logística es multinomial
						  solver = 'lbfgs', # Algoritmo a utilizar en el problema de optimización, aunque es el dado por defecto
						  max_iter = 1000))

log_pipe = Pipeline(steps=[('preprocesador', preprocesador),('clf', log)])

params_log = {'clf__C': [2.0, 1.0, 0.1, 0.01, 0.001]}

# Perceptron
perceptron = (Perceptron(tol = 1e-3, random_state = 0))

perceptron_pipe = Pipeline(steps=[('preprocesador', preprocesador), ('clf', perceptron)])

rc = (RidgeClassifier(normalize=True, random_state=SEED, tol=0.1))

rc_pipe = Pipeline(steps=[('preprocesador', preprocesador),('clf',rc)])

params_rc = {'clf__alpha': [1.0, 0.1, 0.01, 0.001]}

sgd = (SGDClassifier(loss = 'hinge', penalty = 'l2', random_state = SEED))

sgd_pipe = Pipeline(steps=[('preprocesador', preprocesador), ('clf', sgd)])


mejores_clasificadores = []

grid = GridSearchCV(svc_pipe, params_svc, scoring='accuracy', cv=5) # Cross-validation para elegir hiperparámetros
grid.fit(X, y)
mejores_clasificadores.append(grid.best_estimator_)

grid = GridSearchCV(log_pipe, params_log, scoring='accuracy', cv=5) # Cross-validation para elegir hiperparámetros
grid.fit(X, y)
mejores_clasificadores.append(grid.best_estimator_)

grid = GridSearchCV(rc_pipe, params_rc , scoring='accuracy', cv=5) # Cross-validation para elegir hiperparámetros
grid.fit(X, y)
mejores_clasificadores.append(grid.best_estimator_)

mejores_clasificadores.append(perceptron_pipe.fit(X,y))
mejores_clasificadores.append(sgd_pipe.fit(X,y))

accuracy_svc = mejores_clasificadores[0].score(X, y)
accuracy_log = mejores_clasificadores[1].score(X, y)
accuracy_rc = mejores_clasificadores[2].score(X,y)
accuracy_perceptron = mejores_clasificadores[3].score(X,y)
accuracy_sgd = mejores_clasificadores[4].score(X,y)

print('accuracy_svc: {}'.format(accuracy_svc))
print('accuracy_log: {}'.format(accuracy_log))
print('accuracy_rc: {}'.format(accuracy_rc))
print('accuracy_perceptron: {}'.format(accuracy_perceptron))
print('accuracy_sdg: {}'.format(accuracy_sgd))

clasificador = mejores_clasificadores[0]
for i in range(len(mejores_clasificadores)):
	if(mejores_clasificadores[i].score(X,y) > clasificador.score(X,y)):
		clasificador = mejores_clasificadores[i]

clasificador.fit(X, y)
y_predict = clasificador.predict(X_test)

# Matriz de confusion
cm = confusion_matrix(y_test, y_predict)
cm = 100*cm.astype("float64")/cm.sum(axis=1)[:,np.newaxis]
print(cm)
fig = plt.figure()
ax = fig.add_subplot()
cax = ax.matshow(cm, cmap ="BuGn")
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set(title="Matriz de confusión",
         xticks=np.arange(10),
         yticks=np.arange(10),
         xlabel="Etiqueta real",
         ylabel="Etiqueta predicha")

for i in range(10):
	for j in range(10):
		if(i == 0):
			ax.text(j, i,"{:.0f}%".format(cm[i, j]), ha="center", va="top")
		elif(i == 9):
			ax.text(j, i, "{:.0f}%".format(cm[i, j]), ha="center", va="bottom")
		else:
			ax.text(j, i, "{:.0f}%".format(cm[i, j]), ha="center", va="center")

plt.show()


print("E_in: {}".format(1 - clasificador.score(X, y)))
print("E_out: {}".format(1 - clasificador.score(X_test, y_test)))
