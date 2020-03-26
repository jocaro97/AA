# -*- coding: utf-8 -*-

#############################
#####     LIBRERIAS     #####
#############################

import random
import numpy as np

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
def sgd(x, y, lr, max_iters, tam_minibatch):
	w = np.zeros((3,))
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

print ('Bondad del resultado para grad. descendente estocastico:\n')
print ("Ein: ", Err(x, y, w))
print ("Eout: ", Err(x_test, y_test, w))

input("\n--- Pulsar tecla para continuar ---\n")

# Algoritmo Pseudoinversa

w = pseudoinverse(x, y)

print ('\nBondad del resultado para el algoritmo de la pseudoinversa:\n')
print ("Ein: ", Err(x,y,w))
print ("Eout: ", Err(x_test, y_test, w))


#------------------------------Ejercicio 2 -------------------------------------#

# Simula datos en un cuadrado [-size,size]x[-size,size]
def simula_unif(N, d, size):
	return

# EXPERIMENTO
# a) Muestra de entrenamiento N = 1000, cuadrado [-1,1]x[-1,1]

print ('Ejercicio 2\n')
print ('Muestra N = 1000, cuadrado [-1,1]x[-1,1]')




# -------------------------------------------------------------------

# d) Ejecutar el experimento 1000 veces

print ('Errores Ein y Eout medios tras 1000reps del experimento:\n')
print ("Ein media: ", Ein_media)
print ("Eout media: ", Eout_media)

input("\n--- Pulsar tecla para continuar ---\n")
