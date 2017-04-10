# -*- coding: utf-8 -*-

from login import login
from nlp_twitter import *
from nlp_twitter import process, normalize, punctuation, palabras_vacias_es, palabras_vacias_en
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn import cross_validation                            
from sklearn.svm import SVC                                          
from sklearn.linear_model import LogisticRegression                  
from sklearn import metrics 
from nltk import NaiveBayesClassifier, MaxentClassifier, SklearnClassifier
from collections import defaultdict
from operator import itemgetter

api = login()

hashtags = ['#CocaCola', '#GustavoCerati', '#RockArgentino', '#Fernet']  #en este caso vamos a recomendar hashtags

def buscar_hashtags(api, hashtags): #funcion simple para obtener tweets en base a hashtags
    rows = []           
    for hashtag in hashtags:                                           
        for tweet in tweepy.Cursor(api.search,                                                                       
                                   q= hashtag + " filter:links",#busqueda de tweets por hashtag priorizando links                             
                                   rpp = 10,                                                              
                                   include_entities = True,                                                           
                                   lang = "es").items(10):             
            text =  ' '.join(tweet.text.replace('\n', ' ').split()).encode('utf-8')                                     
            rows.append([text, str(hashtag).lower()])       
    return rows

#antes usabamos solamente los textos, ahora queremos saber si que podemos recomendar a los usuarios en la red utilizando los hashtags de las influencias para la recomendacion
def generar_rasgos_recomendaciones_desde_usuarios(textos):
    influencias_train = []
    for i in xrange(len(textos)):
        for j in textos[i]:
            tag = ['#'+word[1:] for word in j.split() if word.startswith('#')]
            if len(tag) == 0:
                pass
            else:
                if len(tag) > 1:
                    tag = tag[0]
                influencias_train.append([j,tag])
    return influencias_train

def generar_lista_hashtags(test):      
    hashtags = []
    for i in test:
        if type(i[1]) == list:
            ht = i[1][0]
        else:
            ht = i[1]
        if not ht in hashtags:
            hashtags.append(ht)
    return hashtags

def limpiar_test(test):           
    for i in xrange(len(test)): 
        if type(test[i][1]) == list:
            test[i][1] = test[i][1][0]
    return test 

def unigrams(tweet):
    features = defaultdict(list)
    words = tweet.split()
    for w in words:
        features[w] = True
    return features

def feature_extractor(tweet):
    return unigrams(tweet)

def features_from_tweets(preprocess): #obtenemos rasgos de cada texto, en nuestro caso utilizaremos unigramas como rasgos
    feature_labels = []                
    for i in range(0, len(preprocess)):               
        features = feature_extractor(preprocess[i][0])     
        feature_labels.append((features, preprocess[i][1]))
    return feature_labels

def extract_features(tweet): #para realizar la recomendacion, tenemos que "tokenizar" los rasgos de cada tweet
    features = defaultdict(list)
    words = tweet.split()
    for w in words:
        features[w] = True
    return features

#feature_labes: resultado de utilizar features_from_tweets con lista de tweets del tipo [texto, hashtag]
def entrenar_recomendacion(feature_labels):
    cv = cross_validation.KFold(len(feature_labels), n_folds=10) #realizamos validacion cruzada para ver como esta funcionando nuestro clasificador, es decir, ver si podemos encontrar alguna configuracion de parametros que nos de un resultado mas exacto
    sum_accuracy = 0                                         
    sum_average_precision = 0                                        
    sum_f1 = 0                                                       
    sum_precision = 0         
    sum_recall = 0 
    sum_roc_auc = 0             
    k = 0 
    for traincv, testcv in cv:
#        classifier = NaiveBayesClassifier.train(feature_labels[traincv[0]:traincv[len(traincv)-1]])
#        classifier = MaxentClassifier.train(feature_labels[traincv[0]:traincv[len(traincv)-1]])
        classifier = SklearnClassifier(SVC(kernel='linear', probability=True)).train(feature_labels[traincv[0]:traincv[len(traincv)-1]]) #elegimos nuestro algoritmo para clasificar, en este caso, una maquina de soporte vectorial para problemas de clasificacion
#        classifier = SklearnClassifier(knn()).train(feature_labels[traincv[0]:traincv[len(traincv)-1]])
        y_true = []
        y_pred = []
        for i in range(len(testcv)):
            y_true.append(feature_labels[testcv[i]][1])
            y_pred.append(classifier.classify(feature_labels[testcv[i]][0]))
        acc = metrics.accuracy_score(y_true, y_pred) #tomamos la exactitud de nuestra clasificacion con los datos de entrenamiento y los de prueba
        sum_accuracy += acc #sumamos la exactitud total
        k += 1
        print(str(k) + ')exactitud: ' + str(acc))
        print('Clases utilizadas: ' + str(y_true))
        print('Predicciones: ' + str(y_pred))
        print('')
    print ('EXACTITUD: ' + str(sum_accuracy/k))
    classifier.train(feature_labels)
    return classifier

#classifier: clasificador para la cual realizar la recomendacion, test: los hashtags que veremos si se pueden recomendar con otros, hashtags: lista de hashtags
def recomendar(classifier, test, hashtags):                                    
    y_true = []                                                      
    y_pred = []                                                      
                                                                     
    #calculamos la exactitud total de la clasificacion para saber si la recomendacion es optima   
    for i in range(len(test)):                                       
        y_true.append(test[i][1]) #guardamos en la lista los hashtags que utilizamos para clasificar                                 
        y_pred.append(classifier.classify(extract_features(test[i][0]))) #guardamos los hashtags que se pueden recomendar
    print('Clases utilizadas: ' + str(y_true))                             
    print('Predicciones: ' + str(y_pred))                        
    print('')                                                        
                                                                     
    total_accuracy = metrics.accuracy_score(y_true, y_pred) #calculamos la exactitud total de la prediccion, si es alta (mayor que 0.5) quiere decir que si hay hashtags que se pueden recomendar con exactitud
    print("Exactitud Total: " + str(total_accuracy))                  
    print('')                                                        
    #calculamos la exactitud top3 de nuestros datos                                                                   
    correctly_classified = 0                                         
    for i in range(len(test)):                                       
        dist = classifier.prob_classify(extract_features(test[i][0]))                                                                     
        predicted_probs = []                                         
        for label in dist.samples(): 
            predicted_probs.append((label, dist.prob(label)))        
        predicted_probs = sorted(predicted_probs, key=itemgetter(1), reverse=True)                                                        
        #si la clase correcta esta entre la exactitud optima, sumamos un punto por clasificacion correcta                                             
        for j in range(3):                                           
            if test[i][1] in predicted_probs[j][0]:                  
                correctly_classified += 1                            
                break                                                
    print("Top3 Exactitud Total: " + str(float(correctly_classified)/float(len(test))))                                                    
    print('') 

    #calculamos la precision de clasificacion para cada hashtag
    for hashtag in hashtags:
        hashtag = hashtag.replace('#', '')
        validation_set_filtered = []
        for row in test:
            if hashtag in row[1] or hashtag.lower() in row[1]:
                validation_set_filtered.append(row)
        y_true_h = []
        y_pred_h = []
        for i in range(len(validation_set_filtered)):
            y_true_h.append(validation_set_filtered[i][1])
            y_pred_h.append(classifier.classify(extract_features(validation_set_filtered[i][0])))
        precision_h = metrics.accuracy_score(y_true_h, y_pred_h)
        print('#' + hashtag + ': precision = ' + str(precision_h) +
              ' (support = ' + str(len(validation_set_filtered)) + ')')
        print('')

    #calculamos la precision Top3 para cada hashtag
    recommend = []
    for hashtag in hashtags:
        hashtag = hashtag.replace('#', '')
        validation_set_filtered = []
        for row in test:
            if hashtag in row[1] or hashtag.lower() in row[1]:
                validation_set_filtered.append(row)
        for i in range(len(validation_set_filtered)):
            dist = classifier.prob_classify(extract_features(validation_set_filtered[i][0]))
            predicted_probs = []
            for label in dist.samples():
                predicted_probs.append((label, dist.prob(label)))
            predicted_probs = sorted(predicted_probs, key=itemgetter(1), reverse=True)
            #if the correct label is in the top3 of our classifier, count as correctly classified
            correctly_classified = 0
            for j in range(3):
                if validation_set_filtered[i][1] in predicted_probs[j][0]:
                    correctly_classified += 1
                    break
        try:                                                         
            print('#' + hashtag + ': top3 precision = ' + str(correctly_classified/len(validation_set_filtered)) +                        
              ' (support = ' + str(len(validation_set_filtered)) + ')')               
            print('')
            recommend.append([hashtag, correctly_classified/len(validation_set_filtered), len(validation_set_filtered)]) 
            print('')
        except ZeroDivisionError:
            print "No hay garantias para recomendar #" + str(hashtag)

    return recommend

#ejemplo, una vez que obtuve una lista de hashtags mas recomendados, ordeno las listas en orden descendiente y obtengo los 10 hashtags mas recomendables
#list(reversed(sorted(rec, key = lambda x: x[2])))[:10]
