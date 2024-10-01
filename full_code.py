import keras
import scikeras
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def build_rna():
    rna = Sequential()
    #! Puesto que tenemos 11 features, añadimos 10 nodos en la capa de entrada
    rna.add(Dense(units = 11, kernel_initializer = "uniform", activation = "relu", input_dim = 11))
    # Añadimos una segunda capa oculta
    rna.add(Dense(units = 6, kernel_initializer = "uniform",  activation = "relu"))
    # Añadimos una primera capa de dropout
    rna.add(Dropout(0.1))
    # Añadimos una tercera capa oculta
    rna.add(Dense(units = 6, kernel_initializer = "uniform",  activation = "relu"))
    # Añadimos una segunda capa de dropout
    rna.add(Dropout(0.2))
    # Añadimos una cuarta capa oculta
    rna.add(Dense(units = 6, kernel_initializer = "uniform",  activation = "relu"))
    # Añadimos una tercera capa de dropout
    rna.add(Dropout(0.3))
    #! Un solo nodo para  la salida y activación sigmoide al ser una clasificación binaria
    rna.add(Dense(units = 1, kernel_initializer = "uniform",  activation = "sigmoid"))
    # Compilamos la RNA y unimos todos los nodos y capas
    rna.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy", "Recall", "Precision"])
    # Devolver la RNA
    return rna


# Preparamos la RNA al conjunto de entrenamiento para poder utilizar el k-fold cv
rna = KerasClassifier(build_fn = build_rna, batch_size = 50, epochs = 100)

# Aplicación del k-fold 
accuracies = cross_val_score(
    estimator=rna, 
    X = x_train, y = y_train, 
    cv = 10, 
    n_jobs=-1, 
    verbose = 1)

# Obtenemos el vector con los resultados de las precisiones
accuracies

# Obtenemos la media y la varianza del promedio de las precisiones
# En cuál de los 4 gráficos sesgo-varianza nos encontramos??
mean = accuracies.mean()
variance = accuracies.std()
print(mean)
print(variance)


# Realización de la predicción

# Entrenamos el modelo
rna.fit(x_train, y_train)

# Hacemos las predicciones sobre los datos de prueba
y_pred = rna.predict(x_test)

# Convertimos las predicciones y los datos reales a un formato adecuado
y_pred = np.array(y_pred).flatten()
y_test = np.array(y_test).flatten()

# Crea la matriz de confusión
cm = confusion_matrix(y_test, y_pred)
print(cm)
# Visualiza la matriz de confusión
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d')
plt.ylabel('Etiqueta verdadera')
plt.xlabel('Etiqueta predicha')
plt.show()


"""
Accuracy y varianza:


Precisión, Recall y F1:


Matriz de confusión:


"""