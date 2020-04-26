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

# Funciones reutilizadas del ejercicio1.py
def signo(x):
	if x >= 0:
		return 1
	return -1

def f(x, y, a, b):
	return signo(y - a*x - b)

def porcentajes(x, y, f):
	signos = y * f(x)
	return 100*len(signos[signos >= 0])/len(y)

# EJERCICIO 2.1: ALGORITMO PERCEPTRON

def ajusta_PLA(datos, labels, max_iter, vini):
    #CODIGO DEL ESTUDIANTE
	w = vini.copy()

	for i in range(max_iter):
		w_ant = w.copy()

		for dato, label in zip(datos, labels):
			res = w.dot(dato)
			if(signo(res) != label):
				w += label*dato

		if(np.all(w == w_ant)):
			return w, i+1

	return w, i+1

#CODIGO DEL ESTUDIANTE

# Generamos los parámetros de la recta
a,b = simula_recta([-50,50])
# Generamos el conjunto de datos
puntos = simula_unif(100, 2, [-50,50])

x = []
y = []

# Construimos el conjutno de datos y las etiquetas
for i in range(len(puntos)):
	# Añadimos un 1 en la primera casilla del vector de características
	x.append(np.array([1, puntos[i,0], puntos[i,1]]))
	# Utilizamos la recta y = ax + b para etiquetar
	y.append(signo(f(puntos[i,0], puntos[i,1], a, b)))

x = np.array(x, np.float64)
y = np.array(y, np.float64)

# SIN RUIDO
# a) Vector 0
w = np.array([0.0,0.0,0.0])
w, i = ajusta_PLA(x, y, 1000, w)
w_fun = lambda x: x.dot(w)

print("SIN RUIDO\n")
print("Vector 0")
print("Iteraciones: {}".format(i))
print("Porcentaje correctos: {}".format(porcentajes(x, y, w_fun)))
print("\n")

# b) Vectores aleatorios
# Random initializations
iterations = []
porcent = []
for i in range(0,10):
	#CODIGO DEL ESTUDIANTE
	w = np.random.rand(3)
	w, i = ajusta_PLA(x, y, 1000, w)
	w_fun = lambda x: x.dot(w)
	iterations.append(i)
	porcent.append(porcentajes(x, y, w_fun))

print("Vetores aleatorios")
print('Valor medio de iteraciones necesario para converger: {}'.format(np.mean(np.asarray(iterations))))
print('Valor medio del porcentaje de aciertos: {}'.format(np.mean(np.asarray(porcent))))

scatter = plt.scatter(x[:,1], x[:,2], c = y)
ymin, ymax = np.min(x[:, 2]), np.max(x[:, 2])
plt.ylim(ymin, ymax)
xmin, xmax = np.min(x[:,1]), np.max(x[:,1])
plt.xlim(xmin,xmax)
legend1 = plt.legend(*scatter.legend_elements(), title = "Clases", loc = "upper right")
plt.plot(x[:,1], (-w[1]*x[:,1] - w[0])/w[2], label = "sgdRL")
plt.plot(x[:,1] , x[:,1]*a + b, label = "recta: ax + b")
plt.xlabel('Coordenada X')
plt.ylabel('Coordenada Y')
plt.legend(loc = "lower right")
plt.gca().add_artist(legend1)
plt.title("Recta de separación y recta obtenida con PLA (vector inicial aleatorio)")
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

# Ahora con los datos del ejercicio 1.2.b
#CON RUIDO
positivos = []
negativos = []
for i in range(len(y)):
	if(y[i] > 0):
		positivos.append(i)
	else:
		negativos.append(i)

# Modificamos la etiqueta del 10% de cada clase
indices_pos = np.random.choice(positivos, int(0.1*len(positivos)), replace = False )
for ind in indices_pos:
	y[ind] = -1

indices_neg = np.random.choice(negativos, int(0.1*len(negativos)), replace = False )
for ind in indices_neg:
	y[ind] = +1

#CODIGO DEL ESTUDIANTE
# a) Vector 0
w = np.array([0.0,0.0,0.0])
w, i = ajusta_PLA(x, y, 1000, w)
w_fun = lambda x: x.dot(w)

print("CON RUIDO\n")
print("Vector 0")
print("Iteraciones: {}".format(i))
print("Porcentaje correctos: {}".format(porcentajes(x, y, w_fun)))
print("\n")

# b) Vectores aleatorios
# Random initializations
iterations = []
porcent = []
for i in range(0,10):
	#CODIGO DEL ESTUDIANTE
	w = np.random.rand(3)
	w, i = ajusta_PLA(x, y, 1000, w)
	w_fun = lambda x: x.dot(w)
	iterations.append(i)
	porcent.append(porcentajes(x, y, w_fun))

print("Vetores aleatorios")
print('Valor medio de iteraciones necesario para converger: {}'.format(np.mean(np.asarray(iterations))))
print('Valor medio del porcentaje de aciertos: {}'.format(np.mean(np.asarray(porcent))))

scatter = plt.scatter(x[:,1], x[:,2], c = y)
ymin, ymax = np.min(x[:, 2]), np.max(x[:, 2])
plt.ylim(ymin, ymax)
xmin, xmax = np.min(x[:,1]), np.max(x[:,1])
plt.xlim(xmin,xmax)
legend1 = plt.legend(*scatter.legend_elements(), title = "Clases", loc = "upper right")
plt.plot(x[:,1], (-w[1]*x[:,1] - w[0])/w[2], label = "sgdRL")
plt.plot(x[:,1] , x[:,1]*a + b, label = "recta: ax + b")
plt.xlabel('Coordenada X')
plt.ylabel('Coordenada Y')
plt.legend(loc = "lower right")
plt.gca().add_artist(legend1)
plt.title("Recta de separación y recta obtenida con PLA (vector inicial aleatorio)")
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 3: REGRESIÓN LOGÍSTICA CON STOCHASTIC GRADIENT DESCENT

def gradRL(dato, label, w):
	return -label*dato/(1 + np.exp(label*w.dot(dato)))

def sgdRL(datos, labels, lr = 0.01):
    #CODIGO DEL ESTUDIANTE
	N, dim = datos.shape
	w = np.array([0.0,0.0,0.0])
	indices = np.arange(N)
	cambio = True

	while cambio:
		w_ant = w.copy()
		indices = np.random.permutation(indices)
		for ind in indices:
			w += -lr * gradRL(datos[ind], labels[ind], w)

		if (np.linalg.norm(w - w_ant) > 0.01):
			cambio = True
		else:
			cambio = False

	return w

# Funcion para calcular el error logistico
def Err(x,y,w):
	return np.mean(np.log(1 + np.exp(-y*x.dot(w))))

#CODIGO DEL ESTUDIANTE
intervalo = [0, 2]
a, b = simula_recta(intervalo)
# Conjuntos de datos
N = 100
datos = simula_unif(N, 2, intervalo)
x = []
# Creamos el vector de características
for i in range(datos.shape[0]):
	x.append(np.array([1, datos[i][0], datos[i][1]]))

x = np.array(x, np.float64)

# Añadimos las etiquetas
y = np.empty((N, ))
for i in range(N):
  y[i] = f(datos[i, 0], datos[i, 1], a, b)

w = sgdRL(x, y)
print("Vector de pesos:")
print(w)
input("\n--- Pulsar tecla para continuar ---\n")

# Usar la muestra de datos etiquetada para encontrar nuestra solución g y estimar Eout
# usando para ello un número suficientemente grande de nuevas muestras (>999).


#CODIGO DEL ESTUDIANTE
# Conjuntos test
N = 1000
datos_test = simula_unif(N, 2, intervalo)
x_test = []
# Creamos el vector de características
for i in range(datos_test.shape[0]):
	x_test.append(np.array([1, datos_test[i][0], datos_test[i][1]]))

x_test = np.array(x_test, np.float64)
# Añadimos las etiquetas
y_test = np.empty((N, ))
for i in range(N):
  y_test[i] = f(datos_test[i, 0], datos_test[i, 1], a, b)


print ("Bondad del resultado para grad. descendente estocastico:\n")
print ("Eout: ", Err(x_test, y_test, w))
# Función solución
f_w = lambda x: w[2]*x[:,2] + w[1]*x[:,1] + w[0]
print ("Porcentaje de aciertos", porcentajes(x_test, y_test, f_w))

input("\n--- Pulsar tecla para continuar ---\n")

# Mostramos los resultados
scatter = plt.scatter(x_test[:,1], x_test[:,2], c = y_test)
ymin, ymax = np.min(x_test[:, 2]), np.max(x_test[:, 2])
plt.ylim(ymin, ymax)
xmin, xmax = np.min(x_test[:,1]), np.max(x_test[:,1])
plt.xlim(xmin,xmax)
legend1 = plt.legend(*scatter.legend_elements(), title = "Clases", loc = "upper right")
plt.plot(x_test[:,1], (-w[1]*x_test[:,1] - w[0])/w[2], label = "sgdRL")
plt.plot(x_test[:,1] , x_test[:,1]*a + b, label = "recta: ax + b")
plt.xlabel('Coordenada X')
plt.ylabel('Coordenada Y')
plt.legend(loc = "lower right")
plt.gca().add_artist(legend1)
plt.title("Recta de separación y recta de sdgLR")
plt.show()
