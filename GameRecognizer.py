import csv
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

def tokenArrayToString(array):
    s = ""
    for i in range(0,len(array)):
        if i == 0:
            s += array[i]
        s += " " + array[i]
    return s

def preprocess(text):
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    sw = stopwords.words("english")
    stemmer = SnowballStemmer("english", ignore_stopwords=True)

    text = text.lower()
    text = nltk.word_tokenize(text)
    for tokens in text:
        if tokens in symbols:
            text.pop(text.index(tokens))
    for tokens in text:
        if tokens in sw:
            text.pop(text.index(tokens))
    for tokens in text:
        if "'" in tokens:
            text.pop(text.index(tokens))
    for tokens in text:
        if len(tokens) == 1:
            text.pop(text.index(tokens))
    for tokens in text:
        text[text.index(tokens)] = stemmer.stem(tokens)

    return text

def handleData(rows):
    game_reviews = []
    game_labels = []
    rows+=1
    i = 0
    with open("AWK.csv", encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=";")
        for row in reader:
            if i == rows:
                break
            reviewText = row[6].replace("&gt", ".")
            gameLabel = row[7]
            game_reviews.append(reviewText)
            game_labels.append(gameLabel)
            i+=1
    game_reviews.pop(0)
    game_labels.pop(0)
    len_data = len(game_reviews)
    for text in game_reviews:
        game_reviews[game_reviews.index(text)] = tokenArrayToString(preprocess(text))
    training_data = game_reviews[:int(len_data*0.9)]
    training_data_labels = game_labels[:int(len_data*0.9)]
    testing_data = game_reviews[int(len_data*0.9):]
    testing_data_labels = game_labels[int(len_data*0.9):]
    return training_data, training_data_labels, testing_data, testing_data_labels


def calculate_TFIDF(corpus):
    vectorizer = TfidfVectorizer()
    vector = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names()
    dense = vector.todense()
    denselist = dense.tolist()
    df = pd.DataFrame(denselist, columns=feature_names)
    return df

def document_features(DataFrame, document_index, labels):
    return (DataFrame.iloc[document_index], labels[document_index])

d = handleData(1000)
#print(d[0])
#print(d[1])
#print(d[2])
#print(d[3])
labels_training = d[1]
labels_testing = d[3]
df_training = calculate_TFIDF(d[0])
print(df_training)
df_testing = calculate_TFIDF(d[2])

training_featureset = [(document_features(df_training,i,labels_training)) for i in range(0,len(d[0]))]
testing_featureset = [(document_features(df_testing,j,labels_testing)) for j in range(0, len(d[2]))]


classifier = nltk.NaiveBayesClassifier.train(training_featureset)
print(nltk.classify.accuracy(classifier, testing_featureset))
classifier.show_most_informative_features(20)
print(classifier.labels())

