#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 14:13:49 2020

@author: kushagra
"""


import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from nltk.tokenize import RegexpTokenizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics


token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)



data = pd.read_csv('RottenTommatoData/train.tsv', sep='\t')

text_counts= cv.fit_transform(data['Phrase'])

X_train, X_test, y_train, y_test = train_test_split(text_counts, data['Sentiment'], test_size=0.01, random_state=1)


clf = MultinomialNB().fit(X_train, y_train)

predicted= clf.predict(X_test)


print("MultinomialNB Accuracy:",metrics.accuracy_score(y_test, predicted))



Testdata = pd.read_csv('RottenTommatoData/test.tsv', sep='\t')
text_counts_test = cv.fit_transform(Testdata['Phrase'])
print (clf.predict(text_counts_test)[81])


'''text = [[1,'nice movie',5]]
tcount = cv.fit_transform(text[0][1])'''
