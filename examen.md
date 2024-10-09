### Posible contenido del examen

-Revisar analisis de riesgo crediticio.
-Revisar e investigar el K-fold - validación cruzada- para cuando no son redes neuronales.
-Hacer el de tumors pero dejando más limpio el código.
-Buscar el de cancer de mama.


1. La parte de preprocesado ya la hicimos con Alan, así que entendí que no hay que darle muchas más vueltas a esto. Pero sí que tendremos que preprocesar los datos para que sean aptos para redes neuronales (vamos, que estén escalados para que la RNA asigne bien los pesos y hacer algo con los categóricos). No creo que nos fuerce a hacer una PCA, ni reducción recursiva, ni cosas raras.

2. Insistió mucho en el dataframe del tumor cerebral (creo que es el que más veces ha mencionado y nos recomendó fuertemente que lo revisásemos antes del examen). También ha mencionado varias veces el de cáncer de mama, el churnmodelling -que bueno, ese es el reto-, y me pareció entender que el examen va a tener una convolucional para imágenes (el tema es que nos ponga una clasificación binaria o que sea multiclase). 

3. Toda la arquitectura de la red neuronal es material de examen: nodos, capas de entrada, ocultas, salida, dropout y densas. Funciones de activacion, optimizadores, algoritmos de entrenamiento, epocs, etc. Vamos, que aunque luego sea copiar y pegar el código, la red neuronal hay que llevarla al dedillo.

4. Métricas de validación y K-Fold cross validation: Las ha mencionado mucho, pero no me ha quedado claro qué tan importante son para él. Gráfica de pérdida, curva de ROC, matriz de confusión, precisión, recall, F1,  accuracy, varianza (acordaros de la gráfica donde ponía la diana con la relación bias/varianza)... Aquí dependiendo de si es regresión o clasificación se usan unas métricas u otras. Con tener el código listo para validar, después es darle a ejecutar y listo.

5. Clustering y k-means: gráfico del codo y de la silueta.


