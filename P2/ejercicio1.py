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

# EJERCICIO 1.1: Dibujar una gráfica con la nube de puntos de salida correspondiente
#CODIGO DEL ESTUDIANTE

# Llamamos a la función con los parametros dados
x = simula_unif(50, 2, [-50,50])
# Mostramos los resultados junto a un cuadrado con esquina inferrior izquierda [-50,50] y longitud y altura  100
plt.plot(x[:,0], x[:,1], "bo", label = "simula_unif")
rect = plt.Rectangle([-50,-50], 100, 100, fill = False)
plt.ylim(-75, 75)
plt.xlim(-75, 75)
plt.xlabel('Coordenada X')
plt.ylabel('Coordenada Y')
plt.title("simula_unif")
plt.gca().add_patch(rect)
plt.legend()
plt.show()

#CODIGO DEL ESTUDIANTE
# Llamamos a la función con los parametros dados
x = simula_gaus(50, 2, np.array([5,7]))
# Mostramos los resultados junto a un cuadrado con esquina inferrior izquierda [-5,7] y longitud 10 y altura  14
plt.plot(x[:,0], x[:,1], "bo", label = "simula_gaus")
rect = plt.Rectangle([-5,-7], 10, 14, fill = False)
plt.ylim(-15, 15)
plt.xlim(-15, 15)
plt.xlabel('Coordenada X')
plt.ylabel('Coordenada Y')
plt.title("simula_gaus")
plt.gca().add_patch(rect)
plt.legend()
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################


# EJERCICIO 1.2: Dibujar una gráfica con la nube de puntos de salida correspondiente

# La funcion np.sign(0) da 0, lo que nos puede dar problemas
def signo(x):
	if x >= 0:
		return 1
	return -1

def f(x, y, a, b):
	return signo(y - a*x - b)

#CODIGO DEL ESTUDIANTE
# Fijamos la semilla para obtener el mismo conjunto en todos los ejercicios
np.random.seed(1)
# Generamos los parámetros de la recta
a,b = simula_recta([-50,50])
# Generamos la nube de puntos
puntos = simula_unif(100, 2, [-50,50])
x = []
y = []
# Construimos el conjunto de datos y las etiquetas
for i in range(len(puntos)):
	x.append(np.array([puntos[i,0], puntos[i,1]]))
	# Utilizamos la recta y = ax + b para etiquetar
	y.append(signo(f(puntos[i,0], puntos[i,1], a, b)))

x = np.array(x, np.float64)
y = np.array(y, np.float64)

# 2 a) Dibujar una gráfica donde los puntos muestren el resultado de su etiqueta, junto
# con la recta usada para ello.
# Mostramos los puntos con sus etiquetas junto a la recta usada para etiquetar
scatter = plt.scatter(x[:,0], x[:,1], c = y)
legend1 = plt.legend(*scatter.legend_elements(), title = "Clases", loc = "upper right")
plt.plot(x[:,0] , x[:,0]*a + b, label = "recta: ax + b")
ymin, ymax = np.min(x[:, 1]), np.max(x[:, 1])
plt.ylim(ymin, ymax)
xmin, xmax = np.min(x[:,0]), np.max(x[:,0])
plt.xlim(xmin,xmax)
plt.xlabel('Coordenada X')
plt.ylabel('Coordenada Y')
plt.legend(loc = "lower right")
plt.gca().add_artist(legend1)
plt.title("Puntos etiquetados junto a la recta utilizada para etiquetar")
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

# 1.2.b. Dibujar una gráfica donde los puntos muestren el resultado de su etiqueta, junto con la recta usada para ello
# Array con 10% de indices aleatorios para introducir ruido

#CODIGO DEL ESTUDIANTE
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

# Mostramos los puntos conr ruido junto con sus etiquetas y la recta utilizada para etiquetar.
scatter = plt.scatter(x[:,0], x[:,1], c = y)
legend1 = plt.legend(*scatter.legend_elements(), title = "Clases", loc = "upper right")
plt.plot(x[:,0], x[:,0]*a + b, label = "recta: ax + b")
ymin, ymax = np.min(x[:, 1]), np.max(x[:, 1])
plt.ylim(ymin, ymax)
xmin, xmax = np.min(x[:,0]), np.max(x[:,0])
plt.xlim(xmin,xmax)
plt.xlabel('Coordenada X')
plt.ylabel('Coordenada Y')
plt.legend(loc = "lower right")
plt.gca().add_artist(legend1)
plt.title("Puntos etiquetados con ruido junto a la recta utilizada para etiquetar")
plt.show()
# Mostrar el porcentaje de MAL clasificados

input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 1.3: Supongamos ahora que las siguientes funciones definen la frontera de clasificación de los puntos de la muestra en lugar de una recta

def plot_datos_cuad(X, y, fz, title='Point cloud plot', xaxis='x axis', yaxis='y axis'):
    #Preparar datos
    min_xy = X.min(axis=0)
    max_xy = X.max(axis=0)
    border_xy = (max_xy-min_xy)*0.01

    #Generar grid de predicciones
    xx, yy = np.mgrid[min_xy[0]-border_xy[0]:max_xy[0]+border_xy[0]+0.001:border_xy[0],
                      min_xy[1]-border_xy[1]:max_xy[1]+border_xy[1]+0.001:border_xy[1]]
    grid = np.c_[xx.ravel(), yy.ravel(), np.ones_like(xx).ravel()]
    pred_y = fz(grid)
    # pred_y[(pred_y>-1) & (pred_y<1)]
    pred_y = np.clip(pred_y, -1, 1).reshape(xx.shape)

    #Plot
    f, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(xx, yy, pred_y, 50, cmap='RdBu',vmin=-1, vmax=1)
    ax_c = f.colorbar(contour)
    ax_c.set_label('$f(x, y)$')
    ax_c.set_ticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
    ax.scatter(X[:, 0], X[:, 1], c=y, s=50, linewidth=2,
                cmap="RdYlBu", edgecolor='white')

    XX, YY = np.meshgrid(np.linspace(round(min(min_xy)), round(max(max_xy)),X.shape[0]),np.linspace(round(min(min_xy)), round(max(max_xy)),X.shape[0]))
    positions = np.vstack([XX.ravel(), YY.ravel()])
    ax.contour(XX,YY,fz(positions.T).reshape(X.shape[0],X.shape[0]),[0], colors='black')

    ax.set(
       xlim=(min_xy[0]-border_xy[0], max_xy[0]+border_xy[0]),
       ylim=(min_xy[1]-border_xy[1], max_xy[1]+border_xy[1]),
       xlabel=xaxis, ylabel=yaxis)
    plt.title(title)
    plt.show()


#CODIGO DEL ESTUDIANTE
def porcentajes(x, y, f):
	signos = y * f(x)
	return 100*len(signos[signos >= 0])/len(y)

# Construimos las funciones necesarias para llamar a la función que muestra  las gráficas porporcionada.
f_0 = lambda x: x[:, 1] - a*x[:, 0] - b
f_1 = lambda x: (x[:, 0] - 10)**2 + (x[:, 1] - 20)**2 - 400
f_2 = lambda x: 0.5*(x[:, 0] + 10)**2 + (x[:, 1] - 20)**2 - 400
f_3 = lambda x: 0.5*(x[:, 0] - 10)**2 - (x[:, 1] + 20)**2 - 400
f_4 = lambda x: x[:, 1] - 20*x[:, 0]**2 - 5*x[:, 0] + 3

# Mostramos los resultados junto con el porcentaje de aciertos.
plot_datos_cuad(x,y,f_0, title = "Recta")
print("Porcentaje de aciertos de la recta: ")
print(porcentajes(x,y,f_0))
plot_datos_cuad(x,y,f_1, title = "Primera elipse")
print("Porcentaje de aciertos de la primera elipse: ")
print(porcentajes(x,y,f_1))
plot_datos_cuad(x,y,f_2, title = "Segunda elipse")
print("Porcentaje de aciertos de la segunda elipse: ")
print(porcentajes(x,y,f_2))
plot_datos_cuad(x,y,f_3, title = "Hiperbola")
print("Porcentaje de aciertos de la hiperbola: ")
print(porcentajes(x,y,f_3))
plot_datos_cuad(x,y,f_4, title = "Parábola")
print("Porcentaje de aciertos de la parábola: ")
print(porcentajes(x,y,f_4))
