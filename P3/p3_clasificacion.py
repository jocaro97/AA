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

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Perceptron, RidgeClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.base import BaseEstimator

import warnings

# --------------------------------------------------------------------------------------
# Quitamos los warnings
warnings.filterwarnings('ignore')
# --------------------------------------------------------------------------------------
# Semilla
SEED = 100
np.random.seed(SEED)

# Clase que funciona como cualquier estimador
class ClfSwitcher(BaseEstimator):
    def __init__(
        self,
        estimator = LogisticRegression(),
    ):
        """
        A Custom BaseEstimator that can switch between classifiers.
        :param estimator: sklearn object - The classifier
        """

        self.estimator = estimator


    def fit(self, X, y=None, **kwargs):
        self.estimator.fit(X, y)
        return self


    def predict(self, X, y=None):
        return self.estimator.predict(X)


    def predict_proba(self, X):
        return self.estimator.predict_proba(X)


    def score(self, X, y):
        return self.estimator.score(X, y)

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

# Calculamos el tamaño de los conjuntos y su porcentaje
print("Hay {} datos de enternamiento".format(X.shape[0]))
print("Hay {} datos de test".format(X_test.shape[0]))

porc_train = 100*X.shape[0]/(X.shape[0]+ X_test.shape[0])
porc_test = 100*X_test.shape[0]/(X.shape[0]+ X_test.shape[0])
print("Hay {}% de datos de entrenamiento y un {}% de datos de test.".format(porc_train,porc_test))

# Comprobamos que no hay ningún valor perdido, también que
# ningún valor se salga del intervalo [0,16] y ninguna etiqueta del intervalo [0.9]
print("Número de valores perdidos en el conjunto de entrenamiento: {}".format(X.isnull().sum().sum()))
print("Número de valores perdidos en el conjunto de test: {}".format(X_test.isnull().sum().sum()))
print("Valor mínimo de las caraterísticas del conjunto de entrenamiento: {}".format(X.values.min()))
print("Valor máximo de las caraterísticas del conjunto de entrenamiento: {}".format(X.values.max()))
print("Valor mínimo de las caraterísticas del conjunto de test: {}".format(X_test.values.min()))
print("Valor máximo de las caraterísticas del conjunto de test: {}".format(X_test.values.max()))
print("Valor mínimo de las etiquetas del conjunto de entrenamiento: {}".format(y_df.values.min()))
print("Valor máximo de las etiquetas del conjunto de entrenamiento: {}".format(y_df.values.max()))
print("Valor mínimo de las etiquetas del conjunto de test: {}".format(y_df.values.min()))
print("Valor máximo de las etiquetas del conjunto de test: {}".format(y_df.values.max()))


# Comprobamos que las clases estan balanceadas
train_df = pd.DataFrame(data = np.c_[X,y])
clases = np.unique(train_df.values[:,-1])
numero_elementos = []
for i in clases:
	numero_elementos.append(y_df['Dígito'].value_counts()[i])

df_plot = pd.DataFrame(columns= ["Dígitos", "Número de ejemplos"], data =[[c,n] for c, n in zip(clases,numero_elementos)])
sns.barplot(x="Dígitos", y ="Número de ejemplos", data = df_plot)
plt.title("Número de ejemplos de cada clase en el conjunto Train")
plt.show()
input("\n--- Pulsar tecla para continuar ---\n")


test_df = pd.DataFrame(data = np.c_[X_test,y_test])
clases = np.unique(test_df.values[:,-1])
numero_elementos = []
for i in clases:
	numero_elementos.append(y_df['Dígito'].value_counts()[i])

df_plot = pd.DataFrame(columns= ["Dígitos", "Número de ejemplos"], data =[[c,n] for c, n in zip(clases,numero_elementos)])
sns.barplot(x="Dígitos", y ="Número de ejemplos", data = df_plot)
plt.title("Número de ejemplos de cada clase en el conjunto Test")
plt.show()
input("\n--- Pulsar tecla para continuar ---\n")


# Preprocesado
preprocesado = [("escalado", StandardScaler()),
                ("PCA", PCA(n_components=0.95))]

preprocesador = Pipeline(preprocesado)

# Mostramos la matriz de correlaciones antes del preprocesado de datos
def mostrar_correlaciones(datos):
	f, ax = plt.subplots(figsize=(10, 8))
	corr = datos.corr()
	sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), square=True, ax=ax)
	f.suptitle('Matriz Correlaciones')
	plt.show()

mostrar_correlaciones(X)
input("\n--- Pulsar tecla para continuar ---\n")

# Mostramos la matriz de correlaciones después del preprocesado de datos
def muestra_correlaciones_procesados(datos):
	f, ax = plt.subplots(figsize=(10, 8))
	corr = np.corrcoef(datos.T)
	sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),square=True, ax=ax)
	f.suptitle('Matriz Correlaciones')
	plt.show()

datos_preprocesados = preprocesador.fit_transform(X)
muestra_correlaciones_procesados(datos_preprocesados)
input("\n--- Pulsar tecla para continuar ---\n")


# Entrenamiento
# Añadimos el clasificador ClfSwitcher para evitar errores de compilación
preprocesado = [("escalado", StandardScaler()),
                ("PCA", PCA(n_components=0.95)),('clf', ClfSwitcher())]

preprocesador = Pipeline(preprocesado)

# Modelos
modelos = [
		  {'clf': [LogisticRegression(penalty='l2', # Regularización Ridge (L2)
		  						  multi_class='ovr', # Indicamos que la regresión logística es multinomial
		  						  solver = 'lbfgs', # Algoritmo a utilizar en el problema de optimización, aunque es el dado por defecto
		  						  max_iter = 1000)],
	       'clf__C':[2.0, 1.0, 0.1, 0.01, 0.001]},
		  {'clf': [Perceptron(penalty = 'l2', # Regularización de Ridge (L2)
                              tol = 1e-3, # Criterio de parada
                              class_weight = "balanced")]},  #clases balanceada
		  {'clf': [RidgeClassifier(normalize=True, # datos normalizados
                                   class_weight = "balanced", # clases balanceadas
                                   random_state=SEED,
                                   tol=0.1)],
		   'clf__alpha': [1.0, 0.1, 0.01, 0.001]},
		  ]

# cross -validation
grid = GridSearchCV(preprocesador, modelos, scoring='accuracy', cv=5, n_jobs = -1)
grid.fit(X, y)
clasificador = grid.best_estimator_
# Mostramos el clasificador elegido
print("Clasificador elegifo: {}".format(clasificador))
y_predict = clasificador.predict(X_test)

# Matriz de confusion
cm = confusion_matrix(y_test, y_predict)
cm = 100*cm.astype("float64")/cm.sum(axis=1)[:,np.newaxis]
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

# Añadimos los porcentajes a las celdas
for i in range(10):
	for j in range(10):
		ax.text(j, i, "{:.0f}%".format(cm[i, j]), ha="center", va="center")

plt.show()
input("\n--- Pulsar tecla para continuar ---\n")

# Resultados
print("E_in: {}".format(1 - clasificador.score(X, y)))
print("E_test: {}".format(1 - clasificador.score(X_test, y_test)))
