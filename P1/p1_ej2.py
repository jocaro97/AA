# -*- coding: utf-8 -*-

#############################
#####     LIBRERIAS     #####
#############################

import random
import numpy as np
import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------#
#---------------------- Ejercicio sobre regresión lineal -----------------------#
#-------------------------------------------------------------------------------#

#------------------------------Ejercicio 1 -------------------------------------#


# Funcion para leer los datos
def readData(file_x, file_y):
	# Leemos los ficheros
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []

	# Solo guardamos los datos cuya clase sea la 1 o la 5
	for i in range(0,datay.size):
		if datay[i] == 5 or datay[i] == 1:
			if datay[i] == 5:
				y.append(1)
			else:
				y.append(-1)
			x.append(np.array([1, datax[i][0], datax[i][1]]))

	x = np.array(x, np.float64)
	y = np.array(y, np.float64)

	return x, y

# Funcion para calcular el error
def Err(x,y,w):
	return (1/len(x))* np.linalg.norm(x.dot(w) - y)**2

# Calcula derivada de error para un modelo de regresión lineal.
def dErr(x, y, w):
  return 2/len(x)*(x.T.dot(x.dot(w) - y))

# Gradiente Descendente Estocastico
def sgd(x, y, lr, max_iters, tam_minibatch, tam = 3):
	w = np.zeros((tam,))
	it = 0
	indices = np.arange(len(x))
	batch_start = 0

	while it < max_iters:
		it = it + 1
		if(batch_start == 0):
			indices = np.random.permutation(indices)

		batch_end = batch_start + tam_minibatch
		ind = indices[batch_start: batch_end]
		w = w - lr * dErr(x[ind, :], y[ind], w)

		batch_start += tam_minibatch
		if(batch_start > len(x)):
			batch_start = 0

	return w

# Algoritmo pseudoinversa
def pseudoinverse(x, y):
	u, s, v = np.linalg.svd(x)
	d = np.diag([0 if np.allclose(p, 0) else 1/p for p in s])
	w = v.T.dot(d).dot(d).dot(v).dot(x.T).dot(y)
	return w

# Lectura de los datos de entrenamiento
x, y = readData("./datos/X_train.npy", "./datos/y_train.npy")
# Lectura de los datos para el test
x_test, y_test = readData("./datos/X_test.npy", "./datos/y_test.npy")

print ('EJERCICIO SOBRE REGRESION LINEAL\n')
print ('Ejercicio 1\n')
# Gradiente descendente estocastico

w = sgd(x, y, 0.01, 20000, 32)
plt.scatter(x[:,1], x[:,2], c = y)
plt.scatter(x[:,1], -w[1]*x[:,1] - (w[0]/w[2]))
plt.show()

print ('Bondad del resultado para grad. descendente estocastico:\n')
print ("Ein: ", Err(x, y, w))
print ("Eout: ", Err(x_test, y_test, w))

input("\n--- Pulsar tecla para continuar ---\n")

# Algoritmo Pseudoinversa

w = pseudoinverse(x, y)
plt.scatter(x[:,1], x[:,2], c = y)
plt.scatter(x[:,1], -w[1]*x[:,1] - (w[0]/w[2]))
plt.show()
print ('\nBondad del resultado para el algoritmo de la pseudoinversa:\n')
print ("Ein: ", Err(x,y,w))
print ("Eout: ", Err(x_test, y_test, w))


#------------------------------Ejercicio 2 -------------------------------------#

# Simula datos en un cuadrado [-size,size]x[-size,size]
def simula_unif(N, d, size):
	return np.random.uniform(-size, size, (N,d))

# Función que asigna las etiquetas
def f_etiq(x_1, x_2):
	return np.sign((x_1 - 0.2)**2 + x_2**2 - 0.6)
# EXPERIMENTO
# a) Muestra de entrenamiento N = 1000, cuadrado [-1,1]x[-1,1]

print ('Ejercicio 2\n')
print ('Muestra N = 1000, cuadrado [-1,1]x[-1,1]')
x_aux = simula_unif(1000, 2, 1)

plt.plot(x_aux, "bo")
plt.show()

def genera_conjunto():
	x_aux = simula_unif(1000, 2, 1)

	# b) Asignamos las etiquetas
	y_f = f_etiq(x_aux[:900, 0], x_aux[:900, 1])
	y_rand = np.random.choice([-1,1], 100)
	y = np.hstack((y_f, y_rand))

	# c) Estimamos los pesos
	x = []
	for i in range(x_aux.shape[0]):
		x.append(np.array([1, x_aux[i][0], x_aux[i][1]]))

	x = np.array(x, np.float64)
	y = np.array(y, np.float64)
	return x , y

x, y = genera_conjunto()

plt.scatter(x[:,1], x[:,2], c = y)

plt.show()

w = sgd(x, y, 0.01, 20000, 32)

plt.scatter(x[:,1], x[:,2], c = y)
plt.scatter(x[:,1], -w[1]*x[:,1] - (w[0]/w[2]))
plt.show()
print ('Bondad del resultado para grad. descendente estocastico:\n')
print ("Ein: ", Err(x, y, w))

input("\n--- Pulsar tecla para continuar ---\n")

# -------------------------------------------------------------------

# d) Ejecutar el experimento 1000 veces

Ein_media = 0
Eout_media = 0

for i in range(1000):
	x_train, y_train = genera_conjunto()
	x_test, y_test = genera_conjunto()

	w = sgd(x_train, y_train, 0.01, 500, 32)
	Ein_media += Err(x_train, y_train, w)
	Eout_media += Err(x_test, y_test, w)

Ein_media /= 1000
Eout_media /= 1000
print ('Errores Ein y Eout medios tras 1000reps del experimento:\n')
print ("Ein media: ", Ein_media)
print ("Eout media: ", Eout_media)

input("\n--- Pulsar tecla para continuar ---\n")

# e) Repetir el mismo experimento anterior pero usando características no lineales.

def aniade_caract(x_aux):
	x = []
	for i in range(x_aux.shape[0]):
		x.append(np.array([1, x_aux[i][1], x_aux[i][2], x_aux[i][1]*x_aux[i][2], x_aux[i][1]**2, x_aux[i][2]**2 ]))

	x = np.array(x, np.float64)

	return x

Ein_media = 0
Eout_media = 0
# REVISAR EL CODIGO
for i in range(1000):
	x, y_train = genera_conjunto()
	x_aux, y_test = genera_conjunto()

	x_train = aniade_caract(x)
	x_test = aniade_caract(x_aux)
	w = sgd(x_train, y_train, 0.01, 500, 32, 6)
	Ein_media += Err(x_train, y_train, w)
	Eout_media += Err(x_test, y_test, w)

Ein_media /= 1000
Eout_media /= 1000
print ('Errores Ein y Eout medios tras 1000reps del experimento con más características:\n')
print ("Ein media: ", Ein_media)
print ("Eout media: ", Eout_media)

input("\n--- Pulsar tecla para continuar ---\n")
