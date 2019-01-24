import pyprind
import pandas as pd
import os

pbar = pyprind.ProgBar(50000)
labels = {'pos':1, 'neg':0}
df = pd.DataFrame()
for s in ('test', 'train'):
	for l in ('pos', 'neg'):
		path = './aclImdb/%s/%s' % (s, l)
		for file in os.listdir(path):
			with open(os.path.join(path, file), 'r') as infile:
				txt = infile.read()
			df = df.append([[txt, labels[l]]], ignore_index=True)
			pbar.update()
df.columns = ['review', 'sentiment']

import numpy as np
np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))
df.to_csv('./movie_data.csv', index=False)

import re
def preprocessor(text):
	text = re.sub('<[^>]*>', '', text)
	emoticons = re.findall('(?::|;|=)(?:-)?(?:\\)|\\(|D|P)', text)
	text = re.sub('[\\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
	return text

df['review'] = df['review'].apply(preprocessor)

text = "Based on an actual story, John Boorman shows the struggle of an American doctor, whose husband and son were murdered and she was continually plagued with her loss. A holiday to Burma with her sister seemed like a good idea to get away from it all, but when her passport was stolen in Rangoon, she could not leave the country with her sister, and was forced to stay back until she could get I.D. papers from the American embassy. To fill in a day before she could fly out, she took a trip into the countryside with a tour guide. ""I tried finding something in those stone statues, but nothing stirred in me. I was stone myself."" <br /><br />Suddenly all hell broke loose and she was caught in a political revolt. Just when it looked like she had escaped and safely boarded a train, she saw her tour guide get beaten and shot. In a split second she decided to jump from the moving train and try to rescue him, with no thought of herself. Continually her life was in danger. <br /><br />Here is a woman who demonstrated spontaneous, selfless charity, risking her life to save another. Patricia Arquette is beautiful, and not just to look at; she has a beautiful heart. This is an unforgettable story. <br /><br />""We are taught that suffering is the one promise that life always keeps."""
print(preprocessor(text))
print(preprocessor("</a>This :) is :( a test :-)!"))
#print(preprocessor(df.loc[0, 'review'][-50:]))

from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
def tokenizer_porter(text):
	return [porter.stem(word) for word in text.split()]
print(tokenizer_porter('runners like running and thus they run'))

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
stop = stopwords.words('english')
print([w for w in tokenizer_porter('a runner likes running and runs a lot')[-10:] if w not in stop])


X_train = df.loc[:25000, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values
X_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].values

from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(strip_accents=None,
	lowercase=False,
	preprocessor=None)
param_grid = [{'vect__ngram_range': [(1,1)],
				'vect__stop_words': [stop, None],
				'vect__tokenizer': [tokenizer, tokenizer_porter],
				'clf__penalty': ['l1', 'l2'],
				'clf__C': [1.0, 10.0, 100.0]},
				{'vect__ngram_range': [(1,1)],
				'vect__stop_words': [stop, None],
				'vect__tokenizer': [tokenizer, tokenizer_porter],
				'vect__use_idf':[False],
				'vect__norm':[None],
				'clf__penalty': ['l1', 'l2'],
				'clf__C': [1.0, 10.0, 100.0]}]

lr_tfidf = Pipeline([('vect', tfidf),('clf', LogisticRegression(random_state=0))])
gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=-1)
gs_lr_tfidf.fit(X_train, y_train)

print('Best programmer set: %s ' % gs_lr_tfidf.best_params_)
print('CV Accuracy: %.3f' % gs_lr_tfidf.best_score_)
clf = gs_lr_tfidf.best_estimator_
print('Test Accuracy: %.3f' % clf.score(X_test, y_test))
