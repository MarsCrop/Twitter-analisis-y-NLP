from nlp_twitter import process, normalize, punctuation, palabras_vacias_es, palabras_vacias_en
from sklearn.feature_extraction.text import CountVectorizer
from sentimiento import intensidad_de_sentimiento
from nltk.sentiment.util import mark_negation
from sklearn.naive_bayes import MultinomialNB # utilizamos la distribucion multinomial para obtener todas las probabilidades posibles, una vez que encontramos mucha probabilidad le asignamos el sentimiento correspondiente

count = CountVectorizer(analyzer="word",                   
                                   tokenizer=lambda text: mark_negation(process(text, stopwords = palabras_vacias_es + palabras_vacias_en)),                        
                                   preprocessor=lambda text: text.replace("<br />", " "),                             
                                   max_features=10000)

def train_Naive_Bayes(textos, matriz, intensidades, start):
    resultados = []
    clf = MultinomialNB()
    clf.fit(matriz, intensidades[start])
    for i in xrange(len(textos)):
        try:
            vectorized = np.resize(count.fit_transform(textos[i]).toarray(), matriz.shape) 
            resultados.append(clf.predict(vectorized))
        except ValueError:
            print "Usuario {} ".format(i) + "no pudo ser ser analizado con Naive Bayes"
            
    print "0 quiere decir que la inclinacion es negativa, 1 quiere decir que la inclinacion es neutral y 2 quiere decir que la inclinacion es positiva"
    return resultados

def write_sentiment_text(textos, results):
    with open("sentimientos.json", "w") as f:
        for i in xrange(len(textos)):
            for j in xrange(len(textos[i])):
                if results[i][j] == 0:
                    json.dump(repr('-'+textos[i][j] + ', neg').encode('utf-8'), f, indent = 4)
                if results[i][j] == 1:
                    json.dump(repr('-'+textos[i][j] + ', neu').encode('utf-8'), f, indent = 4)
                if results[i][j] == 2:
                    json.dump(repr('-'+textos[i][j] + ', pos').encode('utf-8'), f, indent = 4)
