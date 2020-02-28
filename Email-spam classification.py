import pandas as pd
import numpy as np
messages=pd.read_csv('smsspamcollection/SMSSpamCollection',sep='\t',name=["label","messages"])

import re
import nltk
nilt.download('stpwords')

from nltk.corpus import stopwords
from nltk.stem.poter import PorterStemmer
ps=PoterStemmer()
corpus = []
for i in range(0,len(messages)):
	review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
	review = review.lower()
	review = review.split()
	
	review = [ps.stem(word) for word in review if not word in stopwords.word('english')]
	review = ' '.join(review)
	corpus.append(review)


#creating the bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv = Contervectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()

y=pd.get_dummies(messages['label'])
y=y.iloc[:,1].values

from sklearn.model_selection import train_test_selection
X_train, X_test, y_train, y_test = train_terst_split(X,y, test_size=0.20, random_state=0)

#Train the model using bt Navie Bayes Classifier
from sklearn.naive-bayes import MultinominalNB
spam_detect_model = MultinominalNB.fit(X_train, y_train)

y_pred=spam_detect_model.predict(X_test)

from sklearn.mtrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

from sklearn.metrics import accuracy-score
ac = accuracy_score(y_test,y_pred)