from timeline import *
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation as AP
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt

api = login()

T, tweets = timeline(api) #buscamos datos del timeline

T = cleanse_data(T) #limpiamos los datos, es decir, preparar la data para normalizacion. En este caso asumimos que los booleanos se expresan como -1 (None), 0 (False) y 1 (True)

T = np.array([T[0],T[-1]]) #en este caso podemos utilizar dos rasgos, utilizaremos el numero de retweets y si el tweet fue retwitteado o no ya que estos valores son numericos

T[0] = MinMaxScaler().fit_transform(np.float64(T[0])) #como los rasgos de retwitteado o no ya fueron normalizados, procedemos a normalizar la cantidad de retweets para que tenga valores oscilando entre 0 y 1

T = make_2d_mat(T) #tenemos que convertir los vectores a matrices

def kmeans(T, metodo):
    model = KMeans(n_clusters = 2, init=metodo, precompute_distances = True) #el numero de clusters tiene que ser mayor o igual al numero de rasgos que utilizamos, inicializamos con el metodo elegido (k-means++ y random) y establecemos que el algoritmo debe computar las distancias
    model.fit(T) #preparamos la matriz de distancias y la obtencion de centroides
    targets = model.predict(T) #encontramos la posicion estable de todos los datos en sus respectivos clusters
    return targets

#Probamos los cambios en los resultados

print str().join(("Solucion para KMeans sin elegir los puntos de partida", str(kmeans(T,'k-means++'))))
print str().join(("Solucion 1 para KMeans eligiendo aleatoreamente los centroides", str(kmeans(T,'random'))))
print str().join(("Solucion 2 para KMeans eligiendo aleatoreamente los centroides", str(kmeans(T,'random'))))

#visualizamos los resultados
plt.scatter(T[:,0], T[:,1], cmap = kmeans(T,'k-means++'))
plt.show()
plt.scatter(T[:,0], T[:,1], cmap = kmeans(T,'random'))
plt.show()
plt.scatter(T[:,0], T[:,1], cmap = kmeans(T,'random'))
plt.show()

def propagacion_de_afinidad(T):
    model = AP(affinity = 'euclidean') #lo unico que determinamos nosotros es la afinidad, si la afinidad es 'precomputed' debemos introducir en fit nuestra matrix de preferencias en lugar de T
    model.fit(T) #preparamos la matriz de distancias y la obtencion de centroides
    targets = model.predict(T) #encontramos la posicion estable de todos los datos en sus respectivos clusters
    return targets

print  str.join(("Solucion para propagacion de afinidad de los rasgos analizados", str(propagacion_de_afinidad(T))))
