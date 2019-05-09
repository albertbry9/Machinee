import pandas
archivo="weather.nominal.csv"
conjunto_de_datos = pandas.read_csv(archivo, header=0)
X = conjunto_de_datos.iloc[:,0:4].values
y = conjunto_de_datos.iloc[:,4].values
num_instancias, num_caracteristicas = X.shape
print(num_instancias, num_caracteristicas)
print(X, '\n', y)
print("")
#se divide el conjunto de datos en datos de entrenamiento y en datos de prueba
from sklearn.model_selection import train_test_split
#Semilla para generar números aleatorios. Se pone igual a todos para poder comparar los resultados
semilla = 14
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y,
test_size=0.25, random_state=semilla)
print("Hay {} instancias para el entrenamiento".format(y_entrenamiento.shape))
print("Hay {} instancias para las pruebas".format(y_prueba.shape))
print("")

from collections import defaultdict
from operator import itemgetter
def entrenar(X, y_supervisado, caracteristica):
    #Verificamos que la variable contenga un número válido
    num_instancias, num_caracteristicas = X.shape
    assert 0 <= caracteristica < num_caracteristicas
    #Obtenemos todos los valores de dicha característica
    valores = set(X[:,caracteristica])
    #Inicializamos el arreglo de predictores que será retornado
    predictores = dict()
    #Inicializamos el arreglo de errores que será retornado
    errores = []
    for valor_actual in valores:
        clase_mas_frecuente, error = entrenar_con_valor_de_caracteristica(X, y_supervisado, caracteristica, valor_actual)
        predictores[valor_actual] = clase_mas_frecuente
        errores.append(error)
    #Calculamos el error total de usar esta característica para clasificar con OneR
    error_total = sum(errores)
    return predictores, error_total

#Calculamos los predictores basados el el valor de cada carecterística
#y_predicted = np.array([predictors[sample[feature]] for sample in X])
def entrenar_con_valor_de_caracteristica(X, y_supervisado, caracteristica, valor):
    #Creamos un diccionario para contar la frecuencia de cierta característica
    contadores_por_clase = defaultdict(int)
    #Recorremos todas las instancais y contamos la frecuencia de cada par clase/valor
    for instancia, y in zip(X, y_supervisado):
        if instancia[caracteristica] == valor:
            contadores_por_clase[y] += 1
    #Obtenemos la mejor clase, ordenando de forma descendente y escogiendo el primer item
    contadores_por_clase_ordenado = sorted(contadores_por_clase.items(), key=itemgetter(1), reverse=True)
    clase_mas_frecuente = contadores_por_clase_ordenado[0][0]
    #El error es el número de instancias para la cual no se clasifica bien la instancia
    error = sum([contador_por_clase for valor_de_la_clase, contador_por_clase in contadores_por_clase.items() if valor_de_la_clase != clase_mas_frecuente])
    return clase_mas_frecuente, error

#Calculamos todos los predictores
todos_los_predictores = {variable: entrenar(X_entrenamiento, y_entrenamiento, variable) for
variable in range(X_entrenamiento.shape[1])}
errores = {variable: error for variable, (mapping, error) in todos_los_predictores.items()}
mejor_variable, mejor_error = sorted(errores.items(), key=itemgetter(1))[0]
print("El mejor modelo está basado en la variable {0} y tiene como error {1:.2f}".format(mejor_variable, mejor_error))
# Escoger el mejor modelo
modelo = {'variable': mejor_variable,'predictor': todos_los_predictores[mejor_variable][0]}
print(modelo)

import numpy as np
def predecir(X_prueba, modelo):
    variable = modelo['variable']
    predictor = modelo['predictor']
    y_predecida = np.array([predictor[instancia[variable]] for instancia in X_prueba])
    return y_predecida
print("")
y_predecida = predecir(X_prueba, modelo)
print(y_predecida)
exactitud = np.mean(y_predecida == y_prueba) * 100
print("La exactitud es {:.1f}%".format(exactitud))

from sklearn.metrics import classification_report
print(classification_report(y_prueba, y_predecida))