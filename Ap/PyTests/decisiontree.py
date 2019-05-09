import pandas
archivo="iris.data"
columnas=['longitud-sépalo', 'ancho-sépalo', 'longitud-pétalo', 'ancho-pétalo', 'clase']
conjunto_de_datos = pandas.read_csv(archivo, names=columnas)
X = conjunto_de_datos.iloc[:,0:4].values
y = conjunto_de_datos.iloc[:,4].values
print(X)
print(y)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
arbol = DecisionTreeClassifier(criterion="entropy")
modelo=arbol.fit(X, y)
print(modelo)
y_predecido = modelo.predict(X)
print(accuracy_score(y, y_predecido))
print(confusion_matrix(y, y_predecido))
print(classification_report(y, y_predecido))

arbol = DecisionTreeClassifier(criterion="gini")
modelo=arbol.fit(X, y)
print(modelo)
y_predecido = modelo.predict(X)
print(accuracy_score(y, y_predecido))
print(confusion_matrix(y, y_predecido))
print(classification_report(y, y_predecido))