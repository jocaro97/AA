# -*- coding: utf-8 -*-

#############################
#####     LIBRERIAS     #####
#############################

import numpy as np
import math
import matplotlib.pyplot as plt


#-------------------------------------------------------------------------------#
#------------- Ejercicio sobre la búsqueda iterativa de óptimos ----------------#
#-------------------------------------------------------------------------------#


#------------------------------Ejercicio 1 -------------------------------------#
# Constantes
MAX_ITER = 100

# Fijamos la semilla

def E(w):
	return pow(w[0] * np.exp(w[1]) - 2 * w[1] * np.exp(-w[0]), 2)

# Derivada parcial de E respecto de u
def Eu(w):
	return 2 * np.exp(-2 * w[0]) * (w[0] * np.exp(w[0]+w[1]) - 2 * w[1]) * (np.exp(w[0]+w[1]) +2*w[1])

# Derivada parcial de E respecto de v
def Ev(w):
	return 2 * np.exp(-2 * w[0]) * (w[0] * np.exp(w[0]+w[1]) - 2) * (w[0] * np.exp(w[0]+w[1]) -2*w[1])

# Gradiente de E
def gradE(w):
	return np.array([Eu(w), Ev(w)])

def gd(w, lr, grad_fun, fun, epsilon, max_iters = MAX_ITER):
	for it in range(max_iters):
		w = w - lr * grad_fun(w)
		if(fun(w) < epsilon):
			break

	return w, it

print ('\nGRADIENTE DESCENDENTE')
print ('\nEjercicio 1\n')
w, num_ite = gd([1,1], 0.1, gradE, E, pow(10,-14))
print ('Numero de iteraciones: ', num_ite)
input("\n--- Pulsar tecla para continuar ---\n")
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')

input("\n--- Pulsar tecla para continuar ---\n")

#------------------------------Ejercicio 2 -------------------------------------#

def f(w):
	return pow(w[0] - 2, 2) + 2 * pow(w[1] + 2, 2) + 2 * math.sin(2* math.pi * w[0]) * math.sin(2 * math.pi *w[1])

# Derivada parcial de f respecto de x
def fx(w):
	return 2 * (2 * math.pi * math.cos(2 * math.pi * w[0]) * math.sin(2 * math.pi * w[1]) + w[0] - 2)

# Derivada parcial de f respecto de y
def fy(w):
	return 4 * (math.pi * math.sin(2 * math.pi * w[0]) * math.cos(2 * math.pi * w[1]) + w[1] + 2)

# Gradiente de f
def gradf(w):
	return np.array([fx(w), fy(w)])

# a) Usar gradiente descendente para minimizar la función f, con punto inicial (1,1)
# tasa de aprendizaje 0.01 y max 50 iteraciones. Repetir con tasa de aprend. 0.1
def gd_grafica(w, lr, grad_fun, fun, max_iters = MAX_ITER):
	graf = []
	for it in range(max_iters):
		graf.append(fun(w))
		w = w - lr * grad_fun(w)


	plt.plot(range(0,max_iters), graf, 'bo', linestyle='--')
	plt.xlabel('Iteraciones')
	plt.ylabel('f(x,y)')
	plt.show()

print ('Resultados ejercicio 2\n')
print ('\nGrafica con learning rate igual a 0.01')
gd_grafica([1, -1], 0.01, gradf, f, 50)
print ('\nGrafica con learning rate igual a 0.1')
gd_grafica([1, -1], 0.1, gradf, f, 50)
input("\n--- Pulsar tecla para continuar ---\n")


# b) Obtener el valor minimo y los valores de (x,y) con los
# puntos de inicio siguientes:

def gd(w, lr, grad_fun, fun, max_iters = MAX_ITER):
	return w

print ('Punto de inicio: (2.1, -2.1)\n')
print ('(x,y) = (', w[0], ', ', w[1],')\n')
print ('Valor minimo: ',f(w))

input("\n--- Pulsar tecla para continuar ---\n")

print ('Punto de inicio: (3.0, -3.0)\n')
print ('(x,y) = (', w[0], ', ', w[1],')\n')
print ('Valor minimo: ',f(w))

input("\n--- Pulsar tecla para continuar ---\n")

print ('Punto de inicio: (1.5, 1.5)\n')
print ('(x,y) = (', w[0], ', ', w[1],')\n')
print ('Valor minimo: ',f(w))

input("\n--- Pulsar tecla para continuar ---\n")

print ('Punto de inicio: (1.0, -1.0)\n')
print ('(x,y) = (', w[0], ', ', w[1],')\n')
print ('Valor mínimo: ',f(w))

input("\n--- Pulsar tecla para continuar ---\n")
