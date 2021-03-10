import csv
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import matplotlib.pyplot as plt

def tokenArrayToString(array):
    s = ""
    for i in range(0,len(array)):
        if i == 0:
            s += array[i]
        else:
            s += " " + array[i]
    return s

def preprocess(text):
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    sw = stopwords.words("english")
    stemmer = SnowballStemmer("english", ignore_stopwords=True)
    custom_sw = ["game", "games"]

    text = text.lower()
    text = re.sub(r"[^a-zA-Z]+", " ", text)
    text = nltk.word_tokenize(text)

    for tokens in text:
        if tokens in symbols:
            text.pop(text.index(tokens))
    for tokens in text:
        if tokens in sw or tokens in custom_sw:
            text.pop(text.index(tokens))
    for tokens in text:
        if "'" in tokens:
            text.pop(text.index(tokens))
    for tokens in text:
        if len(tokens) == 1:
            text.pop(text.index(tokens))
    for tokens in text:
        if tokens in sw or tokens in custom_sw:
            text.pop(text.index(tokens))
    for tokens in text:
        text[text.index(tokens)] = stemmer.stem(tokens)
    return text

def handleData(rows, classes):
    game_reviews = []
    game_labels = []
    rows+=1
    i = 0
    with open("Bok1.csv", encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=";")
        for row in reader:
            if i == rows:
                break
            reviewText = row[6].replace("&gt", ".")
            gameLabel = row[7]
            if gameLabel in classes:
                game_reviews.append(reviewText)
                game_labels.append(gameLabel)
                i+=1
    game_reviews.pop(0)
    game_labels.pop(0)
    len_data = len(game_reviews)
    for text in game_reviews:
        t = preprocess(text)
        if len(t) < 1:
            game_labels.pop(game_reviews.index(text))
            game_reviews.pop(game_reviews.index(text))
            continue
        game_reviews[game_reviews.index(text)] = tokenArrayToString(t)
    training_data = game_reviews[:int(len_data*0.9)]
    training_data_labels = game_labels[:int(len_data*0.9)]
    testing_data = game_reviews[int(len_data*0.9):]
    testing_data_labels = game_labels[int(len_data*0.9):]
    return training_data, training_data_labels, testing_data, testing_data_labels

def calculate_TFIDF(training_text, testing_text):
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5)
    train = vectorizer.fit_transform(training_text)
    test = vectorizer.transform(testing_text)
    vocabulary = vectorizer.get_feature_names()
    return pd.DataFrame(data=train.toarray(), columns=vocabulary), pd.DataFrame(data=test.toarray(), columns=vocabulary)

def calculate_BoW(training_text, testing_text):
    vectorizer = CountVectorizer()
    train = vectorizer.fit_transform(training_text)
    test = vectorizer.transform(testing_text)
    vocabulary = vectorizer.get_feature_names()
    return pd.DataFrame(data=train.toarray(), columns=vocabulary), pd.DataFrame(data=test.toarray(), columns=vocabulary)

def document_features_TF_IDF(DataFrame, document_index):
    return DataFrame.iloc[document_index]

def document_features_BoW(DataFrameBoW, document_index):
    return DataFrameBoW.iloc[document_index]

def document_features_BoW_TF_IDF(tfidf, bow):
    newdict = {}
    for keys in tfidf.keys():
        newdict[keys] = tfidf[keys]
        newdict["count({})".format(keys)] = bow[keys]
    return newdict

def getLabel(document_index, labels):
    return labels[document_index]

def train_and_test_NBC(d, data):
    #TF-IDF only:
    vectorized_data_1 = calculate_TFIDF(d[0], d[2])
    training_featureset_1 = [(document_features_TF_IDF(vectorized_data_1[0],i),
                              getLabel(i, d[1])) for i in range(0,len(d[0]))
                             ]
    testing_featureset_1 = [(document_features_TF_IDF(vectorized_data_1[1],j),
                             getLabel(j, d[3])) for j in range(0, len(d[2]))
                            ]
    #BoW only:
    vectorized_data_2 = calculate_BoW(d[0], d[2])
    training_featureset_2 = [(document_features_BoW(vectorized_data_2[0], i),
                              getLabel(i, d[1])) for i in range(0, len(d[0]))
                             ]

    testing_featureset_2 = [(document_features_BoW(vectorized_data_2[1], j),
                              getLabel(j, d[3])) for j in range(0, len(d[2]))
                            ]
    #TF-IDF + BoW:

    training_featureset_3 = [(document_features_BoW_TF_IDF(data[0][i], data[2][i]),
                              getLabel(i, d[1])) for i in range(0, len(d[0]))
                              ]
    testing_featureset_3 = [(document_features_TF_IDF(data[1][j], data[3][j]),
                             getLabel(j, d[3])) for j in range(0, len(d[2]))
                            ]

    classifier1 = nltk.NaiveBayesClassifier.train(training_featureset_1)
    classifier2 = nltk.NaiveBayesClassifier.train(training_featureset_2)
    classifier3 = nltk.NaiveBayesClassifier.train(training_featureset_3)

    return round(nltk.classify.accuracy(classifier1, testing_featureset_1) * 100, 1), \
           round(nltk.classify.accuracy(classifier2, testing_featureset_2) * 100, 1), \
           round(nltk.classify.accuracy(classifier3, testing_featureset_3) * 100, 1)

def getData(d):
    vectorized_data1 = calculate_TFIDF(d[0], d[2])
    vectorized_data2 = calculate_BoW(d[0], d[2])

    training_set_TFIDF = []
    testing_set_TFIDF = []
    training_set_BoW = []
    testing_set_BoW = []
    for i in range(0, len(d[0])):
        training_set_TFIDF.append(vectorized_data1[0].iloc[i])
        training_set_BoW.append(vectorized_data2[0].iloc[i])
    for i in range(0, len(d[2])):
        testing_set_TFIDF.append(vectorized_data1[1].iloc[i])
        testing_set_BoW.append(vectorized_data2[1].iloc[i])

    training_set_BoW_TFIDF = []
    testing_set_BoW_TFIDF = []
    for i in range(0, len(d[0])):
        features = []
        features.append(training_set_TFIDF[i])
        features.append(training_set_BoW[i])
        training_set_BoW_TFIDF.append(features)
    for i in range(0, len(d[2])):
        features = []
        features.append(testing_set_TFIDF[i])
        features.append(testing_set_BoW[i])
        testing_set_BoW_TFIDF.append(features)

    arr = np.array(training_set_BoW_TFIDF)
    nsamples, nx, ny = arr.shape
    d2_train_dataset = arr.reshape((nsamples, nx * ny))

    arr2 = np.array(testing_set_BoW_TFIDF)
    nsamples2, nx2, ny2 = arr2.shape
    d2_test_dataset = arr2.reshape((nsamples2, nx2 * ny2))

    return training_set_TFIDF, testing_set_TFIDF, training_set_BoW, testing_set_BoW, d2_train_dataset, d2_test_dataset

def train_and_test_SVM(d, data):

    svm1 = SVC(kernel="linear", C=1)
    svm1.fit(data[0], d[1])
    svm2 = SVC(kernel="linear", C=1)
    svm2.fit(data[2], d[1])
    svm3 = SVC(kernel="linear", C=1)
    svm3.fit(data[4], d[1])

    accuracy1 = svm1.score(data[1], d[3])
    accuracy2 = svm2.score(data[3], d[3])
    accuracy3 = svm3.score(data[5], d[3])

    return round(accuracy1 * 100, 1), \
           round(accuracy2 * 100, 1), \
           round(accuracy3 * 100, 1)

def train_and_test_RFC(d, data):

    rfc1 = RandomForestClassifier(n_estimators=100, bootstrap=True, max_features="sqrt")
    rfc1.fit(data[0], d[1])
    rfc2 = RandomForestClassifier(n_estimators=100, bootstrap=True, max_features="sqrt")
    rfc2.fit(data[2], d[1])
    rfc3 = RandomForestClassifier(n_estimators=100, bootstrap=True, max_features="sqrt")
    rfc3.fit(data[4], d[1])

    accuracy1 = rfc1.score(data[1], d[3])
    accuracy2 = rfc2.score(data[3], d[3])
    accuracy3 = rfc3.score(data[5], d[3])

    return round(accuracy1 * 100, 1), \
           round(accuracy2 * 100, 1), \
           round(accuracy3 * 100, 1)

def main():
    classes = ["PLAYERUNKNOWN'S BATTLEGROUNDS", "Rust", "Grand Theft Auto V", "Dead by Daylight"]
    d = handleData(110, classes)
    data = getData(d)
    NBC1 = train_and_test_NBC(d, data)
    SVM1 = train_and_test_SVM(d, data)
    RFC1 = train_and_test_RFC(d, data)

    d = handleData(120, classes)
    data = getData(d)
    NBC2 = train_and_test_NBC(d, data)
    SVM2 = train_and_test_SVM(d, data)
    RFC2 = train_and_test_RFC(d, data)

    d = handleData(150, classes)
    data = getData(d)
    NBC3 = train_and_test_NBC(d, data)
    SVM3 = train_and_test_SVM(d, data)
    RFC3 = train_and_test_RFC(d, data)

    d = handleData(200, classes)
    data = getData(d)
    NBC4 = train_and_test_NBC(d, data)
    SVM4 = train_and_test_SVM(d, data)
    RFC4 = train_and_test_RFC(d, data)

    d = handleData(300, classes)
    data = getData(d)
    NBC5 = train_and_test_NBC(d, data)
    SVM5 = train_and_test_SVM(d, data)
    RFC5 = train_and_test_RFC(d, data)

    d = handleData(500, classes)
    data = getData(d)
    NBC6 = train_and_test_NBC(d, data)
    SVM6 = train_and_test_SVM(d, data)
    RFC6 = train_and_test_RFC(d, data)

    #0 = TF-IDF, 1 = TF, 2 = TF+TF-IDF

    x = [100, 110, 135, 180, 270, 450]
    NBCy1 = [NBC1[0],NBC2[0],NBC3[0],NBC4[0],NBC5[0],NBC6[0]]
    SVMy1 = [SVM1[0],SVM2[0],SVM3[0],SVM4[0],SVM5[0],SVM6[0]]
    RFCy1 = [RFC1[0],RFC2[0],RFC3[0],RFC4[0],RFC5[0],RFC6[0]]

    NBCy2 = [NBC1[1], NBC2[1], NBC3[1], NBC4[1], NBC5[1], NBC6[1]]
    SVMy2 = [SVM1[1], SVM2[1], SVM3[1], SVM4[1], SVM5[1], SVM6[1]]
    RFCy2 = [RFC1[1], RFC2[1], RFC3[1], RFC4[1], RFC5[1], RFC6[1]]

    NBCy3 = [NBC1[2], NBC2[2], NBC3[2], NBC4[2], NBC5[2], NBC6[2]]
    SVMy3 = [SVM1[2], SVM2[2], SVM3[2], SVM4[2], SVM5[2], SVM6[2]]
    RFCy3 = [RFC1[2], RFC2[2], RFC3[2], RFC4[2], RFC5[2], RFC6[2]]

    plt.figure(1)
    plt.plot(x, NBCy1, label="Naïve Bayes")
    plt.plot(x, SVMy1, label="Support Vector Machine")
    plt.plot(x, RFCy1, label="Random Forest")

    plt.ylabel('Accuracy')
    plt.xlabel('Training samples')

    plt.title('TF-IDF performance')
    plt.legend()
    plt.grid()

    plt.figure(2)
    plt.plot(x, NBCy2, label="Naïve Bayes")
    plt.plot(x, SVMy2, label="Support Vector Machine")
    plt.plot(x, RFCy2, label="Random Forest")

    plt.ylabel('Accuracy')
    plt.xlabel('Training samples')

    plt.title('TF performance')
    plt.legend()
    plt.grid()

    plt.figure(3)
    plt.plot(x, NBCy3, label="Naïve Bayes")
    plt.plot(x, SVMy3, label="Support Vector Machine")
    plt.plot(x, RFCy3, label="Random Forest")

    plt.ylabel('Accuracy')
    plt.xlabel('Training samples')

    plt.title('TF+TF-IDF performance')
    plt.legend()
    plt.grid()


    plt.show()

main()
