#!/usr/bin/python
# -*- coding: utf-8 -*-

from comunidades_egos import *
import operator
import time

def influencia_total(saliente, entrante):
    influencia_impacto = {}
    for i in entrante.keys():
        try:                                               
            influencia_impacto[api.get_user(i).screen_name] = (saliente[i] / entrante[i]) #influencia = saliente/entrante
        except tweepy.error.TweepError:                                                                       
            print "Se ha superado el limite de busquedas, entrando a tiempo de espera"                                            
            time.sleep(60*20)                                                                       
            continue
    return list(reversed(sorted(influencia_impacto.items(), key = operator.itemgetter(1))))

def get_influencias(influencias):
    users = []
    for i in influencias:
        users.append((api.get_user(i[0]).screen_name, i[1]))
    return users

influencia_entrante = nx.katz_centrality_numpy(grafico_egos, weight = lider) #forma de medir la centralidad de autovector, cuanta informacion puede recibir un nodo de otro (e_ji)

influencia_saliente = nx.katz_centrality_numpy(grafico_egos.reverse(), weight = lider) #forma de medir la centralidad de autovector, cuanta informacion puede dar un nodo a otro (e_ij)

nodo_mas_influyente_salida = np.argmax(influencia_saliente.values()) #nodo que mas recibe informacion
nodo_mas_influyente_entrada = np.argmax(influencia_entrante.values()) #nodo que mas envia informacion

"El usuario mas influyente en la salida es " + str(api.get_user(influencia_saliente.items()[nodo_mas_influyente_salida][0]).screen_name)

"El usuario mas influyente en la entrada es " + str(api.get_user(influencia_entrante.items()[nodo_mas_influyente_entrada][0]).screen_name) 

#ordenamos las influencias en orden descendiente
influencia_entrante_descendiente = list(reversed(sorted(influencia_entrante.items(), key = operator.itemgetter(1))))
influencia_saliente_descendiente = list(reversed(sorted(influencia_saliente.items(), key = operator.itemgetter(1))))

#recuperamos los nombres de los usuarios a partir de su id

uinfluence_entrante = get_influencias(influencia_entrante_descendiente)

print uinfluence_entrante

uinfluence_saliente = get_influencias(influencia_saliente_descendiente)

print uinfluence_saliente

#si tomamos los grados de influencia y de salida de cada nodo podemos obtener la influencia como tal, independientemente de la cantidad total de informacion que transmita cada nodo. Asi encontramos una influencia basada en el impacto generado en la red.

influencias = influencia_total(influencia_saliente, influencia_entrante)

print "Nodo mas influyente en la red es {} con {} de impacto".format(influencias[0][0], influencias[0][1])

with open("influencias.txt", 'w') as output_file:
    output_file.write(repr(influencias))

