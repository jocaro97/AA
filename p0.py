import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

# Pone una pausa hasta que se pulse una tecla
def espera():
    input("-------------Pulsa cualquier tecla para continuar.---------------")
    plt.close()

# Parte 1:
# - Lee la base de datos de iris que hay en scikit-learn
# - Obtiene las caracterı́sticas (datos de entrada X) y la clase (y)
# - Se queda con las dos últimas caracterı́sticas (2 últimas columnas de X)
# - Visuliza con un Scatter Plot los datos, coloreando cada clase
# con un color diferente (con rojo, verde y azul), e indicando con
# una leyenda la clase a la que corresponde cada color.
def ej1():
    # Importamos la base de datos Iris
    iris = datasets.load_iris()

    # Mostramos característica y clases
    print("Características: ")
    print(iris.feature_names)

    print("Clases:")
    print(iris.target_names)

    # Nos quedamos con las dos últimas características
    X = iris.data[:, -2:]
    y = iris.target

    colores = {0: 'red', 1: 'green', 2: 'blue'}

    # Visualizar los datos en un scatter plot
    _, ax = plt.subplots()

    # Para cada clase:
    for clase, nombre in enumerate(iris.target_names):
        # Obten los miembros de la clase
        miembros = X[iris.target == clase]

        # Representa en scatter plot
        ax.scatter(miembros[:, 0], miembros[:, 1], c=colores[clase], label=nombre)
        # Añadimos la leyenda

    ax.legend()

    # Visualizamos la gráfica
    plt.show()

# Parte 2:
# - Separa en training (80 % de los datos) y test (20 %) aleatoriamente
# conservando la proporción de elementos en cada clase tanto en
# training como en test.
def ej2():
    # Importamos la base de datos Iris
    iris = datasets.load_iris()
    # Cargamos los datos
    X = iris.data
    y = iris.target

    # Separamos en 80% training y 20% test aleatoriamente
    # stratify = y hace que se mantenga la proporcion de elementos en cada clase
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, shuffle = True, stratify = y)

    # Mostramos el número de elementos de cada conjunto para comprobar
    # que se ha separado correctamente los conjuntos
    print("Número de ejemplos en la base de datos Iris: ")
    print(X.shape[0])
    print("Número de elementos en el conjunto de entrenamiento: ")
    print(X_train.shape[0])
    print("Número de elementos en el conjunto de test: ")
    print(X_test.shape[0])

    # Comprobamos que las proporciones de elementos de cada clase se mantiene
    for clase, nombre in enumerate(iris.target_names):
        proporcion = len(y[y == clase]) / y.shape[0]
        train_prop = len(y_train[y_train == clase]) / y_train.shape[0]
        test_prop = len(y_test[y_test == clase]) / y_test.shape[0]

        print("Proporción de la clase", nombre)
        print("Base de datos Iris", proporcion)
        print("Conjunto training", train_prop)
        print("Conjunto test", test_prop)

# Parte 3:
# - Obtiene 100 valores equiespaciados entre 0 y 2pi
# - Obtiene el valor de sin(x), cos(x) y sin(x)+cos(x) para los 100 valores anteriores
# - Visualiza las tres curvas anteriores con lineas discontinuas en negro, azul y rojo
def ej3():
    # Obtenemos 100 valores equiespaciados entre 0 y 2pi
    x = np.linspace(0, 2 * np.pi, 100)

    # Obtenemos el valor de sin(x) y cos(x) para posteriormente construir las tres funciones
    seno = np.sin(x)
    coseno = np.cos(x)

    # Construimos la gráfica de las tres funciones con los requisitos pedidos
    plt.plot(x, seno, 'k', linestyle = "--", label = "sin(x)")
    plt.plot(x, coseno, 'b', linestyle = "--", label = "cos(x)")
    plt.plot(x, seno + coseno, 'r', linestyle = "--", label = "sin(x) + cos(x)")
    plt.legend()
    plt.show()

# Función principal que ejecuta todos los ejercicios
def main():
    ej1()
    espera()
    ej2()
    espera()
    ej3()
    espera()

main()
