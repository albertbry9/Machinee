import pandas
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

archivo="iris.data"
columnas=['longitud-sépalo', 'ancho-sépalo', 'longitud-pétalo', 'ancho-pétalo', 'clase']
conjunto_de_datos = pandas.read_csv(archivo, names=columnas)
X = conjunto_de_datos.iloc[:,0:4].values
y = conjunto_de_datos.iloc[:,4].values

modelo = KNeighborsClassifier(n_neighbors = 3)
modelo.fit(X,y)
kfold = KFold(n_splits = 10)

resultado = cross_val_score(modelo,X,y, cv= kfold, scoring="accuracy")
print("Accuracy: ", resultado.mean())
resultado = cross_val_score(modelo,X,y,cv = kfold,scoring="f1_weighted")
print("F1: ", resultado.mean())