import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load the CSV data into a pandas dataframe
data = pd.read_csv('conversational_english.csv')

# Separate the text and labels into separate lists
texts = data['text'].tolist()
labels = data['label'].tolist()

# Create a CountVectorizer object to tokenize and count the words in the texts
vectorizer = CountVectorizer()

# Fit the vectorizer to the texts to create the vocabulary and transform the texts into a matrix of word counts
X = vectorizer.fit_transform(texts)

# Convert the labels into a numpy array
y = np.array(labels)

# Train a MultinomialNB classifier on the vectorized texts and labels
nb = MultinomialNB()
nb.fit(X, y)

# Dump the trained MultinomialNB classifier and CountVectorizer object to a pickle file
with open('model.pkl', 'wb') as f:
    pickle.dump((nb, vectorizer), f)