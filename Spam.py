# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 23:12:27 2020

@author: A703569
"""

#Data Preprocesing and EDA

import pandas as pd

Messages_DF = pd.read_csv('SMSSpamCollection',sep='\t',names=["label", "message"])


import re
import nltk
nltk.download('stopwords')


from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
ps = PorterStemmer()
Lemi = WordNetLemmatizer()

corpus = []

for i in range(0, len(Messages_DF)):
    review = re.sub('[^a-zA-Z]', ' ', Messages_DF['message'][i])
    review = review.lower()
    review = review.split()
    review = [Lemi.lemmatize(word) for word in review if word not in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)


##  Creating Bag of words model

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=5000);
X = cv.fit_transform(corpus).toarray()


y = pd.get_dummies(Messages_DF['label'])
y = y.iloc[:,1].values


##Model 
    
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y , test_size = 0.2, random_state = 0)    


from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train,y_train)

y_pred = spam_detect_model.predict(X_test)


from sklearn.metrics import confusion_matrix
confusion_m = confusion_matrix(y_test, y_pred)


from sklearn.metrics import accuracy_score
accuracy = accuracy_score( y_test, y_pred)
