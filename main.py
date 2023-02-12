import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

os.system("cls")
# Load the data from a CSV file
data = pd.read_csv("conversational_english.csv")

# Split the data into training and testing sets
train_data = data[:int(len(data)*0.8)]
test_data = data[int(len(data)*0.8):]

# Extract the features from the text data using CountVectorizer
vectorizer = CountVectorizer()
train_features = vectorizer.fit_transform(train_data['text'])
test_features = vectorizer.transform(test_data['text'])

# Train a Multinomial Naive Bayes classifier on the training data
nb = MultinomialNB()
nb.fit(train_features, train_data['label'])

# Continuously get user input and make predictions
while True:
    user_input = input("Enter a conversational text: ")
    if user_input == "exit":
        break
    input_features = vectorizer.transform([user_input])
    prediction = nb.predict(input_features)[0]
    print("Prediction:", prediction)
    user_confirm = input("Is the prediction correct? (yes/no) ")
    if user_confirm == "no" or user_confirm == "No" or user_confirm == "n" or user_confirm == "N":
        if user_input in data['text'].values:
            continue
        else:
            correct_label = input("Enter the correct label: ")
            data = data.append({'text': user_input, 'label': correct_label}, ignore_index=True)
            # Re-train the classifier with the updated data
            train_data = data[:int(len(data)*0.8)]
            test_data = data[int(len(data)*0.8):]
            train_features = vectorizer.fit_transform(train_data['text'])
            test_features = vectorizer.transform(test_data['text'])
            nb = MultinomialNB()
            nb.fit(train_features, train_data['label'])
            # Save the updated data to the CSV file
            data.to_csv("conversational_english.csv", index=False)
    if user_confirm == "yes" or user_confirm == "Yes" or user_confirm == "y" or user_confirm == "Y":
        if user_input in data['text'].values:
            continue
        else:
            correct_label = prediction
            data = data.append({'text': user_input, 'label': correct_label}, ignore_index=True)
            # Re-train the classifier with the updated data
            train_data = data[:int(len(data)*0.8)]
            test_data = data[int(len(data)*0.8):]
            train_features = vectorizer.fit_transform(train_data['text'])
            test_features = vectorizer.transform(test_data['text'])
            nb = MultinomialNB()
            nb.fit(train_features, train_data['label'])
            # Save the updated data to the CSV file
            data.to_csv("conversational_english.csv", index=False)
    if user_confirm == "cancel":
        continue
    