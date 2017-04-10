# -*- coding: utf-8 -*-

from nlp_twitter import process, normalize, punctuation, palabras_vacias_es, palabras_vacias_en
from nltk.sentiment import SentimentIntensityAnalyzer as sid
from nltk import word_tokenize                                         
from nltk.sentiment.util import mark_negation                 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.base import TransformerMixin
import numpy as np
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    import md5, sha
                                                                       
clf = Pipeline([   
    ('vectorizer', CountVectorizer(analyzer="word",
                                   ngram_range=(1, 2),
                                   tokenizer=lambda text: mark_negation(process(text, stopwords = palabras_vacias_es + palabras_vacias_en)),                        
                                   preprocessor=lambda text: text.replace("<br />", " "),                             
                                   max_features=10000) ),    ('classifier', LinearSVC(class_weight = 'balanced'))])

def intensidad_de_sentimiento(textos): #encontrar si el sentimiento expresado en el tweet es negativo, neutral o positivo (polaridad)
    intensidad = [[] for i in xrange(len(textos))]
    for texto in xrange(len(tokens)):
        for tweet in tokens[texto]:
            print(tweet)
            try:
                ss = sid().polarity_scores(' '.join(tweet)) #ademas devuelve 'compound' (fuerza del sentimiento) que da a conocer la fuerza con la que aparece la polaridad
            except UnicodeDecodeError:
                tokens[texto].remove(tweet)
                pass
            for k in sorted(ss):
                print('{0}: {1}, '.format(k, ss[k]))
            if np.argmax(ss.values()[:3]) == 0:
                intensidad[texto].append(0)
            if np.argmax(ss.values()[:3]) == 1:
                intensidad[texto].append(1)
            if np.argmax(ss.values()[:3]) == 2:
                intensidad[texto].append(2)
    return intensidad

def train_linear_SVM(textos, intensidades, start):
    resultados = []
    svc = LinearSVC()
    try:
        clf.fit(textos[start], intensidades[start])
    except ValueError:
        raise Exception("Ntextos tiene que ser = a Nintensidades")
    print clf.named_steps['vectorizer'].get_feature_names()
    for i in xrange(len(textos)):
        try:
            resultados.append(clf.predict(textos[i]))
            print clf.score(textos[i][:len(intensidades[i])], intensidades[i])
        except ValueError:
            print "Usuario {} ".format(i) + "no pudo ser ser analizado con MSV Lineal"
            
    print "0 quiere decir que la inclinacion es negativa, 1 quiere decir que la inclinacion es neutral y 2 quiere decir que la inclinacion es positiva"
    return resultados



