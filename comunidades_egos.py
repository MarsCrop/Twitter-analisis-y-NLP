import tweepy
import numpy as np
from login import login
from collections import defaultdict, Counter
import networkx as nx
import random
import community
import matplotlib.pyplot as plt
from itertools import permutations

api = login()

def get_users(api, ids):
    t = defaultdict(list) 
    w = []                                            
    for i in xrange(len(ids)):                                     
        pages = tweepy.Cursor(api.followers_ids, id = ids[i]).pages()   
        n_foll = api.get_user(ids[i]).followers_count
        w.append(n_foll)
        try:                                                           
            p = pages.next()                                                                       
            with open('followers.txt', 'w') as f: #abrimos el archivo       
                count = 0                                                                       
                while p:                                               
                    try:                                               
                        for ps in p:                                   
                            t[ids[i]].append(ps)                                                                    
                        p = pages.next()                               
                        count += 1                                                                       
                    except tweepy.error.TweepError:                                                                       
                        print "Se ha superado el limite de busquedas, entrando a tiempo de espera"                                            
                        time.sleep(60*20)                                                                       
                        continue                                     
        except Exception, e: 
            print e                                        
            continue 
    return t, w      

ids = ['XXXXXX',XXXXXX','XXXXXX','XXXXXX','XXXXXXXX','XXXXXXXX','XXXXXXX','XXXXXXX','XXXXXXX','XXXXXX'] #agregar 10 usuarios que tengan algun follower en comun

t, w = get_users(api, ids)

def create_nodes(t):                                                   
    links = []     #comenzamos la busqueda de vinculos entre los resultados (tiene que haber al menos un usuario que este siguiendo a los usuarios
    counter = 0                                                        
    for i in ids:                                                      
        for j in xrange(len(t[i])):                                    
            if [t[i][j], counter] not in links:                        
                links.append([t[i][j], counter])                       
        counter += 1                                                   
                                                                       
    common_foll = Counter(np.array(links)[:,0]).most_common() #comenzamos a buscar followers en comun         
                                                                       
    c_f = []                                                           
    for i in xrange(len(common_foll)):                                 
        if common_foll[i][1] != 1:                                                                       
            c_f.append(common_foll[i][0])                              
                                                                       
    common_links = []                                                  
    for i in xrange(len(np.array(links)[:,0])):                        
        if np.array(links)[:,0][i] in c_f:                                                                       
            common_links.append(np.array(links)[i])

    unqs = np.unique(np.array(common_links)[:,0], return_counts = True, return_index = True) #a estos followers en comun los tenemos que ordenar en base a las personas que siguen

    common_links = np.array(common_links)

    nodes = [] #una vez que tomamos las personas comenzamos a formar los nodos
    for i in xrange(len(common_links)):
        for j in xrange(len(unqs[1])):
            if common_links[i][0] == common_links[unqs[1][j]][0]:
                nodes.append([common_links[i][0], common_links[i][1], common_links[unqs[1]][j][1]])

    imag_nodes = []
    for i in nodes:
        imag_nodes.append(np.unique(i))

    real_nodes = []
    for i in imag_nodes:
        if len(i) < 3:
            pass
        else:
            real_nodes.append(i) #hay nodos mientras hayan usuarios siguiendo mas de un usuario o mientras un usuario sea seguido por varios (+ de uno)

    return np.array(real_nodes) 

real_nodes = create_nodes(t)

def edges(nodes_list):                                                 
    edges = []                                                         
    for i in xrange(len(real_nodes)):                                  
        if (real_nodes[i-1][0] == real_nodes[i][0]) or (real_nodes[i-1][1] == real_nodes[i][1]):                                              
            edges.append((real_nodes[i-1][2], real_nodes[i][2]))                                                                       
    return tuple(edges)

def generar_nodos(ejes): #tenemos que tomar los vertices como nodos para que la visualizacion sea la correcta, es decir, asumir que cada usuario es un nodo
    nodos = []  
    for eje in ejes:       
        if not eje[0] in nodos:
            nodos.append(eje[0])
        if not eje[1] in nodos:
            nodos.append(eje[1])
    return tuple(nodos)  

ejes = edges(real_nodes)

nodos = generar_nodos(ejes)

def grafico_dirigido(ejes, nodos):
    G = nx.DiGraph() #iniciamos un grafico dirigido, es decir, donde la simetria entre todos los vertices de los nodos esta quebrada
    G.add_nodes_from(nodos) #agregamos los nodos (usuarios)
    G.add_edges_from(ejes) #agregamos los vinculos correspondientes, entendidos como ejes

    return G

G = grafico_dirigido(ejes, nodos) #creamos un grafico dirigido y obtenemos una semilla

lider = G.degree().keys()[np.argmax(G.degree().values())] #buscamos el lider a partir del nodo con mayor nodo en el grafico dirigido

print ("Ego para analizar es: " + str(api.get_user(lider).screen_name))

grafico_egos = nx.ego_graph(G, lider, center = True, radius = len(ejes)/len(ids)) #creamos un grafico de egos tomando el lider como centro

def get_comunidades(grafico):
    comunidades = list(nx.k_clique_communities(grafico.to_undirected, 3, nx.find_cliques(grafico.to_undirected()))) #si todo salio bien, los sets de miembros son solamente de ids de usuario sin numeros enteros agregados
    g = nx.Graph() #ahora volvemos a crear un grafico para visualizar todo

    for i in comunidades:
        g.add_nodes_from(i)

    for i in comunidades:
        g.add_edges_from(list(permutations(i,2)))

    pos = nx.spring_layout(g)
    plt.axis('off')
    nx.draw_networkx(g, pos)
    plt.show()

    return comunidades


def get_red_de_egos(grafico):
    pos = nx.spring_layout(grafico) #generamos el grafico en si, con los nodos direccionados forzadamente
    plt.axis('off')
    nx.draw_networkx(grafico, pos)
    plt.show()

def get_nombres_miembros(miembros):     
    members = [[] for i in xrange(len(miembros))]
    for i in xrange(len(miembros)):
        usuarios = api.lookup_users(user_ids = miembros[i])
        for u in usuarios:
            members[i].append(u.screen_name)
        print ("En el grupo formado " + str(i) + " estan " + str(members[i])) #recuperamos los nombres de los usuarios
    return members

members = get_comunidades(G)

print "Miembros de las Comunidades"

name_m = get_nombres_miembros(members)

with open('comunidades.txt' ,'w') as f:     
    f.write(repr(name_m))

get_red_de_egos(grafico_egos)

"Cruces en las Redes de Egos"

print ("Coeficiente de Clustering en la red de egos: " + str(nx.average_clustering(nx.Graph(grafico_egos))))

centralidad = get_nombres_miembros(grafico_egos.edges())

with open('egos.txt' ,'w') as f:     
    f.write(repr(centralidad))
