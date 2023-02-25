import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Load data from CSV file
data = pd.read_csv("conversational_english.csv")

# Create CountVectorizer object to transform text into feature vectors
vectorizer = CountVectorizer()

# Convert text data into feature vectors
X = vectorizer.fit_transform(data["text"])

# Get corresponding labels
y = data["label"]

# Train a Multinomial Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X, y)

# Save the trained model to a pickle file
with open("new.pkl", "wb") as f:
    pickle.dump(clf, f)
