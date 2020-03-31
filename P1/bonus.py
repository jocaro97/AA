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

# Constantes
MAX_ITER = 50

#------------------------------BONUS 1 -------------------------------------#

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

# Derivada con respecto a x dos veces de f(x, y).
def hfxx(x, y):

    return (2 - 8 * math.pi **  2 * math.sin(2 * math.pi * y) * math.sin(2 * math.pi * x))

# Derivada con respecto a y dos veces de f(x, y).
def hfyy(x, y):

    return (4 - 8 * math.pi ** 2 * math.sin(2 * math.pi * x) * math.sin(2 * math.pi * y))

# Derivada cruzada de la función f(x,y).
def hfxy(x, y):

    return 8 * math.pi ** 2 * math.cos(2 * math.pi * y) * math.cos(2 * math.pi * x)

# Matriz hessiana de la función f(x, y).
def hf(x, y):

    return np.array([[hfxx(x, y), hfxy(x, y)],
        [hfxy(x, y), hfyy(x, y)]])

# Función que ejecuta el algoritmo de Newton guardando los puntos para
# posteriormente dibujarlos en una gráfica.
def newton_grafica(w, lr, grad_fun, fun, max_iters = MAX_ITER):
	graf = []
	for it in range(max_iters):
		graf.append(fun(w))
		w = w - lr * np.linalg.inv(hf(w[0],w[1])) @ grad_fun(w)


	plt.plot(range(0,max_iters), graf, "bo" , linestyle='--')
	plt.xlabel('Iteraciones')
	plt.ylabel('f(x,y)')
	plt.show()

# a) Usar el método de Newton para conseguir puntos óptimos la función f, con punto inicial (1,1)
# tasa de aprendizaje 0.01 y max 50 iteraciones. Repetir con tasa de aprend. 0.1
print ('Resultados ejercicio 2\n')
print ('\nGrafica con learning rate igual a 0.01')
newton_grafica([1, -1], 0.01, gradf, f, 50)
print ('\nGrafica con learning rate igual a 0.1')
newton_grafica([1, -1], 0.1, gradf, f, 50)
input("\n--- Pulsar tecla para continuar ---\n")

# Algoritmo de Newton
def newton(w, lr, grad_fun, fun, max_iters = MAX_ITER):
	for it in range(max_iters):
		w = w - lr * np.linalg.inv(hf(w[0],w[1])) @ grad_fun(w)
	return w

# b) Obtener el punto óptimo y los valores de (x,y) con los
# puntos de inicio siguientes y un lr = 0.1:
print("Tasa de aprendizaje 0.1")
print ('Punto de inicio: (2.1, -2.1)\n')
w = newton([2.1, -2.1], 0.1, gradf,f)
print ('(x,y) = (', w[0], ', ', w[1],')\n')
print ('Valor del óptimo obtenido: ',f(w))

input("\n--- Pulsar tecla para continuar ---\n")

print ('Punto de inicio: (3.0, -3.0)\n')
w = newton([3.0, -3.0], 0.1, gradf,f)
print ('(x,y) = (', w[0], ', ', w[1],')\n')
print ('Valor del óptimo obtenido: ',f(w))

input("\n--- Pulsar tecla para continuar ---\n")

print ('Punto de inicio: (1.5, 1.5)\n')
w = newton([1.5, 1.5], 0.1, gradf,f)
print ('(x,y) = (', w[0], ', ', w[1],')\n')
print ('Valor del óptimo obtenido: ',f(w))

input("\n--- Pulsar tecla para continuar ---\n")

print ('Punto de inicio: (1.0, -1.0)\n')
w = newton([1.0, -1.0], 0.1, gradf,f)
print ('(x,y) = (', w[0], ', ', w[1],')\n')
print ('Valor del óptimo obtenido: ',f(w))

input("\n--- Pulsar tecla para continuar ---\n")

# b) Obtener el punto óptimo y los valores de (x,y) con los
# puntos de inicio siguientes y un lr = 0.01:
print("Tasa de aprendizaje 0.01")
print ('Punto de inicio: (2.1, -2.1)\n')
w = newton([2.1, -2.1], 0.01, gradf,f)
print ('(x,y) = (', w[0], ', ', w[1],')\n')
print ('Valor del óptimo obtenido: ',f(w))

input("\n--- Pulsar tecla para continuar ---\n")

print ('Punto de inicio: (3.0, -3.0)\n')
w = newton([3.0, -3.0], 0.01, gradf,f)
print ('(x,y) = (', w[0], ', ', w[1],')\n')
print ('Valor del óptimo obtenido: ',f(w))

input("\n--- Pulsar tecla para continuar ---\n")

print ('Punto de inicio: (1.5, 1.5)\n')
w = newton([1.5, 1.5], 0.01, gradf,f)
print ('(x,y) = (', w[0], ', ', w[1],')\n')
print ('Valor del óptimo obtenido: ',f(w))

input("\n--- Pulsar tecla para continuar ---\n")

print ('Punto de inicio: (1.0, -1.0)\n')
w = newton([1.0, -1.0], 0.01, gradf,f)
print ('(x,y) = (', w[0], ', ', w[1],')\n')
print ('Valor del óptimo obtenido: ',f(w))

input("\n--- Pulsar tecla para continuar ---\n")
print("Gráfico en el que desciende la función.")
newton_grafica([2.1, -2.1], 1, gradf, f, 50)
