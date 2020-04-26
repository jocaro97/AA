# -*- coding: utf-8 -*-
"""
TRABAJO 2
Nombre Estudiante: Johanna Capote Robayna
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


# Fijamos la semilla
np.random.seed(1)

def simula_unif(N, dim, rango):
	return np.random.uniform(rango[0],rango[1],(N,dim))

def simula_gaus(N, dim, sigma):
    media = 0
    out = np.zeros((N,dim),np.float64)
    for i in range(N):
        # Para cada columna dim se emplea un sigma determinado. Es decir, para
        # la primera columna se usará una N(0,sqrt(5)) y para la segunda N(0,sqrt(7))
        out[i,:] = np.random.normal(loc=media, scale=np.sqrt(sigma), size=dim)

    return out


def simula_recta(intervalo):
    points = np.random.uniform(intervalo[0], intervalo[1], size=(2, 2))
    x1 = points[0,0]
    x2 = points[1,0]
    y1 = points[0,1]
    y2 = points[1,1]
    # y = a*x + b
    a = (y2-y1)/(x2-x1) # Calculo de la pendiente.
    b = y1 - a*x1       # Calculo del termino independiente.

    return a, b


#BONUS: Clasificación de Dígitos


# Funcion para leer los datos
def readData(file_x, file_y, digits, labels):
	# Leemos los ficheros
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []
	# Solo guardamos los datos cuya clase sea la digits[0] o la digits[1]
	for i in range(0,datay.size):
		if datay[i] == digits[0] or datay[i] == digits[1]:
			if datay[i] == digits[0]:
				y.append(labels[0])
			else:
				y.append(labels[1])
			x.append(np.array([1, datax[i][0], datax[i][1]]))

	x = np.array(x, np.float64)
	y = np.array(y, np.float64)

	return x, y

# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy', [4,8], [-1,1])
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy', [4,8], [-1,1])


#mostramos los datos
fig, ax = plt.subplots()
ax.plot(np.squeeze(x[np.where(y == -1),1]), np.squeeze(x[np.where(y == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x[np.where(y == 1),1]), np.squeeze(x[np.where(y == 1),2]), 'o', color='blue', label='8')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TRAINING)')
ax.set_xlim((0, 1))
plt.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(np.squeeze(x_test[np.where(y_test == -1),1]), np.squeeze(x_test[np.where(y_test == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x_test[np.where(y_test == 1),1]), np.squeeze(x_test[np.where(y_test == 1),2]), 'o', color='blue', label='8')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TEST)')
ax.set_xlim((0, 1))
plt.legend()
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

#LINEAR REGRESSION FOR CLASSIFICATION

#CODIGO DEL ESTUDIANTE
# Funciones reutilizadas de ejercicio1.py
def signo(x):
	if x >= 0:
		return 1
	return -1

def porcentajes(x, y, f):
	signos = y * f(x)
	return len(signos[signos >= 0])/len(y)

# Error de clasificacion
def Err_clasific(x, y, w):
	f_w = lambda x: w[2]*x[:,2] + w[1]*x[:,1] + w[0]
	return 1 - porcentajes(x, y, f_w)

# Algoritmo pseudoinversa
def pseudoinverse(x, y):
	u, s, v = np.linalg.svd(x)
	d = np.diag([0 if np.allclose(p, 0) else 1/p for p in s])
	w = v.T.dot(d).dot(d).dot(v).dot(x.T).dot(y)
	return w

w_lineal = pseudoinverse(x,y)
print("Modelos de Regresión Lienal")
print(w_lineal)
print ('Bondad del resultado para RL:\n')
print ("Ein: ", Err_clasific(x, y, w_lineal))
print ("Etest: ", Err_clasific(x_test, y_test, w_lineal))

input("\n--- Pulsar tecla para continuar ---\n")

#POCKET ALGORITHM

#CODIGO DEL ESTUDIANTE
def PLA_pocket(datos, labels, max_iter, vini):
    #CODIGO DEL ESTUDIANTE
	w = vini.copy()
	w_mejor = w.copy()
	error_mejor = Err_clasific(datos, labels, w_mejor)

	for i in range(max_iter):

		for dato, label in zip(datos, labels):
			res = w.dot(dato)
			if(signo(res) != label):
				w += label*dato

		error = Err_clasific(datos, labels, w)
		if(error < error_mejor):
			w_mejor = w.copy()
			error_mejor = error

	return w_mejor


w_pla = PLA_pocket(x,y, 1000, w_lineal)

# Mostramos los resultados sobre el conjunto de entrenamiento
scatter = plt.scatter(x[:,1], x[:,2], c = y)
ymin, ymax = np.min(x[:, 2]), np.max(x[:, 2])
plt.ylim(ymin, ymax)
legend1 = plt.legend(*scatter.legend_elements(), title = "Clases", loc = "upper right")
plt.plot(x[:,1], (-w_lineal[1]*x[:,1] - w_lineal[0])/w_lineal[2], label = "RL")
plt.plot(x[:,1], (-w_pla[1]*x[:,1] - w_pla[0])/w_pla[2], label = "RL+PLA")
plt.xlabel('Coordenada X')
plt.ylabel('Coordenada Y')
plt.legend(loc = "lower right")
plt.gca().add_artist(legend1)
plt.title("Grafico sobre los datos de entrenamiento")
plt.show()

# Mostramos los resultados sobre el conjunto de test
scatter = plt.scatter(x_test[:,1], x_test[:,2], c = y_test)
ymin, ymax = np.min(x[:, 2]), np.max(x[:, 2])
plt.ylim(ymin, ymax)
legend1 = plt.legend(*scatter.legend_elements(), title = "Clases", loc = "upper right")
plt.plot(x_test[:,1], (-w_lineal[1]*x_test[:,1] - w_lineal[0])/w_lineal[2], label = "RL")
plt.plot(x_test[:,1], (-w_pla[1]*x_test[:,1] - w_pla[0])/w_pla[2], label = "RL+PLA")
plt.xlabel('Coordenada X')
plt.ylabel('Coordenada Y')
plt.legend(loc = "lower right")
plt.gca().add_artist(legend1)
plt.title("Grafico sobre los datos de test")
plt.show()

print ('Bondad del resultado para RL + PLA:\n')
print ("Ein: ", Err_clasific(x, y, w_pla))
print ("Etest: ", Err_clasific(x_test, y_test, w_pla))


input("\n--- Pulsar tecla para continuar ---\n")
#COTA SOBRE EL ERROR

#CODIGO DEL ESTUDIANTE
# Calculo de la cota del error
def cota(error, N, M, tolerancia):
	return error + np.sqrt(1/(2*N)*(np.log(2 * M /tolerancia)))

ein = Err_clasific(x, y, w_pla)
etest = Err_clasific(x_test, y_test, w_pla)

print ("Cota superior de Eout con Ein: ", cota(ein, len(x), 2**(64*3), 0.05))
print ("Cota superior de Eout con Etest: ", cota(etest, len(x_test), 1, 0.05))
