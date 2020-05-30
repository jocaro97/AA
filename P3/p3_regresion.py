# -*- coding: utf-8 -*-

#############################
#####     LIBRERIAS     #####
#############################

import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, Lasso
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
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
class EstSwitcher(BaseEstimator):
    def __init__(
        self,
        estimator = LinearRegression(),
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

# Leemos los ficheros
atributos = np.array(["state", "county", "community",     "communityname", "fold", "population", "householdsize", "racepctblack",       "racePctWhite", "racePctAsian", "racePctHisp", "agePct12t21",      "agePct12t29", "agePct16t24", "agePct65up", "numbUrban", "pctUrban",        "medIncome", "pctWWage", "pctWFarmSelf", "pctWInvInc", "pctWSocSec",       "pctWPubAsst", "pctWRetire", "medFamInc", "perCapInc", "whitePerCap",       "blackPerCap", "indianPerCap", "AsianPerCap", "OtherPerCap", "HispPerCap",    "NumUnderPov", "PctPopUnderPov", "PctLess9thGrade", "PctNotHSGrad",        "PctBSorMore", "PctUnemployed", "PctEmploy", "PctEmplManu", "PctEmplProfServ",       "PctOccupManu", "PctOccupMgmtProf", "MalePctDivorce", "MalePctNevMarr",        "FemalePctDiv", "TotalPctDiv", "PersPerFam", "PctFam2Par", "PctKids2Par",     "PctYoungKids2Par", "PctTeen2Par", "PctWorkMomYoungKids", "PctWorkMom",        "NumIlleg", "PctIlleg", "NumImmig", "PctImmigRecent", "PctImmigRec5",        "PctImmigRec8", "PctImmigRec10", "PctRecentImmig", "PctRecImmig5",      "PctRecImmig8", "PctRecImmig10", "PctSpeakEnglOnly", "PctNotSpeakEnglWell",   "PctLargHouseFam", "PctLargHouseOccup", "PersPerOccupHous",      "PersPerOwnOccHous", "PersPerRentOccHous", "PctPersOwnOccup",      "PctPersDenseHous", "PctHousLess3BR", "MedNumBR", "HousVacant",       "PctHousOccup", "PctHousOwnOcc", "PctVacantBoarded", "PctVacMore6Mos",        "MedYrHousBuilt", "PctHousNoPhone", "PctWOFullPlumb", "OwnOccLowQuart",        "OwnOccMedVal", "OwnOccHiQuart", "RentLowQ", "RentMedian", "RentHighQ",        "MedRent", "MedRentPctHousInc", "MedOwnCostPctInc",       "MedOwnCostPctIncNoMtg", "NumInShelters", "NumStreet", "PctForeignBorn",        "PctBornSameState", "PctSameHouse85", "PctSameCity85", "PctSameState85",        "LemasSwornFT", "LemasSwFTPerPop", "LemasSwFTFieldOps",      "LemasSwFTFieldPerPop", "LemasTotalReq", "LemasTotReqPerPop",       "PolicReqPerOffic", "PolicPerPop", "RacialMatchCommPol", "PctPolicWhite",     "PctPolicBlack", "PctPolicHisp", "PctPolicAsian", "PctPolicMinor",        "OfficAssgnDrugUnits", "NumKindsDrugsSeiz", "PolicAveOTWorked",        "LandArea", "PopDens", "PctUsePubTrans", "PolicCars", "PolicOperBudg",        "LemasPctPolicOnPatr", "LemasGangUnitDeploy", "LemasPctOfficDrugUn",        "PolicBudgPerPop","ViolentCrimesPerPop", ])

datos = pd.read_csv("./datos/communities.data", header = None, names = atributos)

# Quitamos los atributos no predictivos
datos = datos.drop(columns = ['state','county','community','communityname','fold'])

datos = datos.replace("?", np.nan)
datos_perdidos = datos.columns[datos.isnull().any()]

# Comprobamos que no hayan valores perdidos
print("Número de valores perdidos en el conjunto de datos: {}".format(datos.isnull().sum().sum()))
print("Número de columnas con valores perdidos: {}".format(len(datos_perdidos)))

# Borramos las columnas con valores perdidos
datos.dropna(axis = 'columns', inplace = True)

y = datos.iloc[:, -1]
X = datos.iloc[:, :-1]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20)

# Preprocesado
preprocesado = [("escalado", StandardScaler()), ("PCA", PCA(n_components=0.95))]

preprocesador = Pipeline(preprocesado)

# Mostramos la matriz de correlaciones antes del preprocesado de datos
def mostrar_correlaciones(datos):
	f, ax = plt.subplots(figsize=(10, 8))
	corr = datos.corr()
	sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), square=True, ax=ax)
	f.suptitle('Matriz Correlaciones')
	plt.show()

mostrar_correlaciones(X_train)
input("\n--- Pulsar tecla para continuar ---\n")


# Mostramos la matriz de correlaciones después del preprocesado de datos
def muestra_correlaciones_procesados(datos):
	f, ax = plt.subplots(figsize=(10, 8))
	corr = np.corrcoef(datos.T)
	sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),square=True, ax=ax)
	f.suptitle('Matriz Correlaciones')
	plt.show()

datos_preprocesados = preprocesador.fit_transform(X_train)
muestra_correlaciones_procesados(datos_preprocesados)
input("\n--- Pulsar tecla para continuar ---\n")


# Entrenamiento
# Añadimos el estimador EstSwitcher para evitar errores de compilación
preprocesado = [("escalado", StandardScaler()), ('est', EstSwitcher())]

preprocesador = Pipeline(preprocesado)

# Modelos
modelos = [
		  {'est': [LinearRegression()]},
		  {'est': [SGDRegressor(max_iter=1000, random_state = SEED)],
		   'est__alpha':[1.0, 0.1, 0.01, 0.001, 0.0001]},
		  {'est': [Ridge(max_iter=1000, random_state = SEED)],
		   'est__alpha':[1.0, 0.1, 0.01, 0.001, 0.0001]},
          {'est': [Lasso(max_iter=1000, random_state = SEED)],
           'est__alpha':[1.0, 0.1, 0.01, 0.001, 0.0001]}
		   ]

# cross-validation
grid = GridSearchCV(preprocesador, modelos, scoring='r2', cv=5)
grid.fit(X_train,y_train)
predictor = grid.best_estimator_
# Mostramos el mejor estimador
print("Clasificador elegifo: {}".format(predictor))

predictor.fit(X_train, y_train)

y_predict = predictor.predict(X_test)

# Resultados
print("R^2: {:.4f}".format(r2_score(y_test, y_predict)))
print("E_test: {:.4f}".format(1 - r2_score(y_test, y_predict)))
