# -*- coding: utf-8 -*-

import tweepy
from login import login
from nltk import ngrams, FreqDist
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from string import punctuation
from nltk import AffixTagger, UnigramTagger, BigramTagger
from nltk.collocations import BigramAssocMeasures as bigram 
from nltk.collocations import BigramCollocationFinder as bigram_finder
from nltk.collocations import TrigramAssocMeasures as trigram 
from nltk.collocations import TrigramCollocationFinder as trigram_finder
import string                                      
from nltk import word_tokenize                                   
from nltk.corpus import stopwords, treebank                     
from nltk import BigramTagger, UnigramTagger, AffixTagger
import matplotlib.pyplot as plt
import re
import time

TAGS_TO_KEEP = ['JJ'] #para analisis solamente utilizaremos los sustantivos, los verbos, adjetivos y los adverbios
FREQ_THRESHOLD = 5
FREQ_INTERVAL = 5000
FREQ_LIST = []

api = login()

t0 = AffixTagger(train=treebank.tagged_sents())
t1 = UnigramTagger(train=treebank.tagged_sents(), backoff=t0)
t2 = BigramTagger(train=treebank.tagged_sents(), backoff=t1)

#recolectamos los 10 últimos tweets de los usuarios más influyentes
def recolectar_tweets(influencias):
    tweets = []                   
    for i in xrange(len(influencias)):
        try:                                                           
            tweets.append(api.user_timeline(screen_name = influencias[i][0], count = 30, include_rts = True))
        except tweepy.TweepError:                                      
            print "El usuario {} no admite recoleccion de sus tweets".format(influencias[i][0])
            continue #si el usuario no permite recolectar sus tweets pasamos al siguiente usuario
    return tweets

def text_from_tweets(tweets):             
    textos = [[] for i in xrange(len(tweets))]
    for i in xrange(len(tweets)):
        for j in tweets[i]:     
            textos[i].append(j.text) #de la recoleccion de tweets solamente precisamos el texto
    return textos 

def process(text, tokenizer=TweetTokenizer(), stopwords=[]):
    """Process the text of a tweet:
    - Lowercase
    - Tokenize
    - Stopword removal
    - Digits removal

    Return: list of strings
    """
    text = text.lower()
    tokens = TweetTokenizer(strip_handles = True, reduce_len = True).tokenize(text)
    tokens = normalize(tokens)
    return [tok for tok in tokens if tok not in stopwords and not tok.isdigit()]

#incluimos en token_map todas las contracciones que conozcamos para normalizar los textos
def normalize(tokens):
    token_map = {
        "i'm": "i am",
        "you're": "you are",
        "it's": "it is",
        "we're": "we are",
        "we'll": "we will",
        "ud": "usted",
        "uds": "ustedes",
        "hs": "horas",
        "palante": "para adelante",
        "patras": "para atras",
        "tambin": "tambien",
        "qu": "que",
        "maana": "mañana",
        "adems": "además",
        "contina": "continúa",
    }
    for tok in tokens:
        if tok in token_map.keys():
            for item in token_map[tok].split():
                yield item
        else:
            yield tok

def is_empty(tweet):
    return tweet in ''

def process_all(train):
    processed = []
    for i in train:
        sin_tags = remove_hashtags(i[0])
        sin_usuario = remove_user_tags(sin_tags)
        sin_html = remove_html_entities(sin_usuario)
        alfabeto = remove_punctuation_deep(sin_html)
        tokenizar = process(alfabeto, stopwords = palabras_vacias_es + palabras_vacias_en)
        sin_apostrofe = remove_apostrophes(tokenizar)
        part_of_speech = pos_tag_filter(str(sin_apostrofe), train, t2)
        if not is_empty(part_of_speech):
            processed.append(part_of_speech)
    return processed

def tokens_de_texto(textos):                                          
    tokens = [[] for i in xrange(len(textos))]
    for i in xrange(len(textos)):
        for j in textos[i]:
            try:                                               
                tokens[i].append(process(j, stopwords = palabras_vacias_es + palabras_vacias_en)) #tokenizamos el texto filtrando las palabras vacias en español y en ingles
            except tweepy.error.TweepError:                                                                      
                print "Se ha superado el limite de busquedas, entrando a tiempo de espera"                                            
                time.sleep(60*20)                                                                       
                continue
    return tokens

#ejemplo, busqueda de palabras mas repetidas
def frecuencias_terminos(tokens):                       
    term_freq = FreqDist()
    for i in xrange(len(tokens)):
        for j in tokens[i]:          
            term_freq.update(FreqDist(j))
    y = [count for tag, count in term_freq.most_common(30)]
    x = range(1, len(y)+1)
    print term_freq.most_common(30)
    plt.bar(x, y)
    plt.title("Frecuencias de los terminos")
    plt.ylabel("Frecuencia")
    plt.show()

def tag_to_keep(tag):
    for t in TAGS_TO_KEEP:
            if t in tag:
                return True
    return False

def remove_multiple_spaces(tweet):
    return ' '.join(tweet.split())

def filter_symbols(char):
    if ord(char) in range(65, 91) or ord(char) in range(97, 123) or char == ' ' or char == '\'':
        return char
    else:
        return ''

def apostrophe_filter(char):
    if not char == '\'':
        return char
    else:
        return ''

def remove_apostrophes(tweet):
    return filter(apostrophe_filter, tweet)

def remove_punctuation_deep(tweet):
    return filter(filter_symbols, tweet)

def remove_html_entities(tweet):
    word_list = tweet.split()
    words_to_keep = []
    for word in word_list:
        if not word.startswith('&'):
            words_to_keep.append(word)
    return ' '.join(words_to_keep)

def remove_hashtags(tweet):
    word_list = tweet.split()  
    words_to_keep = []
    for word in word_list:
        if not word.startswith('#'):
            words_to_keep.append(word)
    return ' '.join(words_to_keep)

def remove_user_tags(tweet):
    word_list = tweet.split()
    words_to_keep = []
    for word in word_list:
        if not word.startswith('@'):
            words_to_keep.append(word)
    return ' '.join(words_to_keep)

def is_frequent(word, data, index):
    count = 0
    for i in range(index, FREQ_INTERVAL):
        tweet = data[i][0]
        for w in tweet:
            if word == w:
                count += 1
                break
        if count >= FREQ_THRESHOLD:
            return True
        if i >= (len(data) - 1):
            break
    return False

def is_frequent_word(word, data, index):
    if word in FREQ_LIST:
        return True
    elif is_frequent(word, data, index):
        FREQ_LIST.append(word)
        return True
    return False

def pos_tag_filter(tweet, data, tagger):
    tagged_tweet = tagger.tag(word_tokenize(tweet))
    words_to_keep = []
    for i in range(len(tagged_tweet)):
        tagged_word = tagged_tweet[i]
        word = tagged_word[0]
        tag = tagged_word[1]
        if tag is not None:
            if tag_to_keep(tag):
                words_to_keep.append(word)
        elif is_frequent_word(word, data, i):
                words_to_keep.append(word.lower())
    return ' '.join(words_to_keep)

def process_all_textos(textos):
    processed = [[] for i in xrange(len(textos))]
    for i in xrange(len(textos)):
        for j in textos[i]:
            sin_tags = remove_hashtags(j)
            sin_usuario = remove_user_tags(sin_tags)
            sin_html = remove_html_entities(sin_usuario)
            alfabeto = remove_punctuation_deep(sin_html)
            tokenizar = process(alfabeto, stopwords = palabras_vacias_es + palabras_vacias_en)
            sin_apostrofe = remove_apostrophes(tokenizar)
            if not is_empty(str(sin_apostrofe)):
                processed[i].append(sin_apostrofe)
    return processed

def bigrama(tokens):     
    bigram_medidas = bigram()
    for i in xrange(len(tokens)):
        for j in tokens[i]:                 
            finder = bigram_finder.from_words(j)
            finder.apply_freq_filter(1) #filtramos los bigramas que hayan aparecido una vez          
            print finder.nbest(bigram_medidas.pmi, 30)
            time.sleep(3) #esperamos tres segundos para leer el proximo bigrama

def trigrama(tokens):     
    trigram_medidas = trigram()
    for i in xrange(len(tokens)):
        for j in tokens[i]:                 
            finder = trigram_finder.from_words(j)
            finder.apply_freq_filter(1) #filtramos los trigramas que hayan aparecido una vez          
            print finder.nbest(trigram_medidas.pmi, 30)
            time.sleep(3) #esperamos tres segundos para leer el proximo trigrama

def n_grama(tokens, n):
    for i in xrange(len(tokens)):
        for j in tokens[i]:                 
            ngrama = ngrams(j, n) #generamos frases hechas de n palabras
            for m in ngrama:
                print m
                time.sleep(3)

puntuacion = list(punctuation) #necesitamos una lista de los signos de puntuacion para tokenizar y limpiar el texto
palabras_vacias_es = stopwords.words('spanish') + puntuacion + ['palabravacia'] #eliminamos las palabras vacias, aquellas que no tienen significado
palabras_vacias_en = stopwords.words('english') + puntuacion + ['palabravacia'] #eliminamos las palabras vacias, aquellas que no tienen significado

#tweets = recolectar_tweets(influencias)
#textos = text_from_tweets(tweets)
#tokens = process_all_textos(textos)

#print frecuencias_terminos(tokens)

#bigrama(tokens)
#trigrama(tokens)
#n_grama(tokens, 5) #en este caso buscamos frases hechas de 5 palabras

