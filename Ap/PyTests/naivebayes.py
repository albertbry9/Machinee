import pandas
archivo="iris.data"
columnas=['longitud-sépalo', 'ancho-sépalo', 'longitud-pétalo', 'ancho-pétalo', 'clase']
conjunto_de_datos = pandas.read_csv(archivo, names=columnas)
X = conjunto_de_datos.iloc[:,0:4].values
y = conjunto_de_datos.iloc[:,4].values
print(X)
print(y)

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
gnb = GaussianNB()
modelo = gnb.fit(X, y)
y_predecido = modelo.predict(X)
print(accuracy_score(y, y_predecido))
print(classification_report(y, y_predecido))

X_ejemplo=[(5.1,3.5,1.4,0.2)]
y_predecido = modelo.predict(X_ejemplo)
print(y_predecido)
