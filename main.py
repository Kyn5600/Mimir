import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import os
import warnings
import pickle

warnings.filterwarnings("ignore", category=FutureWarning)

os.system("cls")

# Load the data from a CSV file
data = pd.read_csv("conversational_english.csv")

# Function to train and save the model and vectorizer
def train_and_save_model(data):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data['text'])
    y = data['label']
    clf = MultinomialNB()
    clf.fit(X, y)
    with open('conversational_english_classifier.pkl', 'wb') as f:
        pickle.dump(clf, f)
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

# Initial training
train_and_save_model(data)



# Continuously get user input and make predictions
while True:
    # Load the trained Naive Bayes classifier and vectorizer
    with open('conversational_english_classifier.pkl', 'rb') as f:
        nb = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
        
    user_input = input("Enter a conversational text: ")
    if user_input.lower() == "exit":
        break
    
    input_features = vectorizer.transform([user_input])
    prediction = nb.predict(input_features)[0]
    print("Prediction:", prediction)

    user_confirm = input("Is the prediction correct? (yes/no) ")
    if user_confirm in ["no", "No", "n", "N"]:
        correct_label = input("Enter the correct label: ")
        new_row = pd.DataFrame({'text': [user_input], 'label': [correct_label]})
        data = pd.concat([data, new_row], ignore_index=True)
        data.to_csv("conversational_english.csv", index=False)
        train_and_save_model(data)  # Retrain and save the model and vectorizer
        # Reload the model and vectorizer
        with open('conversational_english_classifier.pkl', 'rb') as f:
            nb = pickle.load(f)
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        print("Retrained model")

    elif user_confirm in ["yes", "Yes", "y", "Y"]:
        # You can decide if you want to do something when the prediction is correct
        continue

    elif user_confirm == "cancel":
        continue
