import pandas
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

archivo="iris.data"
columnas=['longitud-sépalo', 'ancho-sépalo', 'longitud-pétalo', 'ancho-pétalo', 'clase']
conjunto_de_datos = pandas.read_csv(archivo, names=columnas)
X = conjunto_de_datos.iloc[:,0:4].values
y = conjunto_de_datos.iloc[:,4].values

X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X,y,test_size = 0.66)

gnb = GaussianNB()
modelo = gnb.fit(X_entrenamiento,y_entrenamiento)
y_predecido = modelo.predict(X_prueba)
print("Navie Bayes")
print(accuracy_score(y_prueba,y_predecido))
print(classification_report(y_prueba,y_predecido))
print("")
print("Arbol de desicion")
arbol = DecisionTreeClassifier(criterion = "entropy")
modelo = arbol.fit(X_entrenamiento,y_entrenamiento)
y_predecido = modelo.predict(X_prueba)
print(accuracy_score(y_prueba,y_predecido))
print(classification_report(y_prueba,y_predecido))
print("")
print("Validacion Cruzada")
kfold = KFold(n_splits = 10)
resultado = cross_val_score(modelo,X,y,cv = kfold,scoring="accuracy")
print("Accuracy: ", resultado.mean())
resultado = cross_val_score(modelo,X,y,cv = kfold,scoring="f1_weighted")
print("F1: ", resultado.mean())