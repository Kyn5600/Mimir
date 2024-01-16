import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import random 
import os
import warnings
import pickle
warnings.filterwarnings("ignore", category=FutureWarning)

os.system("cls")
# Labels:
# joke_request joke regque regresponse greeting farewell feeling_question feeling_reponse

# Load the data from the CSV file
data = pd.read_csv("conversational_english.csv")

training_data = data[:int(0.8 * len(data))]
testing_data = data[int(0.8 * len(data)):]

# Convert the text into numerical feature vectors using CountVectorizer
vectorizer = CountVectorizer()
text_features = vectorizer.fit_transform(training_data['text'])

# Train a Naive Bayes classifier on the training data

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
       
train_and_save_model(data) 

correct = ''
last_input = ''
other_last_input = ''
last_label = ''
name = "Bot:"
print("Would you like to name me?")
nameCheck = input("You: ")
if nameCheck == 'yes':
    name = input("What is my name? ")
    name = name + ":"
    os.system('cls')
elif nameCheck == 'no':
    os.system('cls')
    print(name,"Okay. Lets continue.")
# Continuously get user input and generate a response based on the label
while True:
    with open('conversational_english_classifier.pkl', 'rb') as f:
        nb = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    
    if last_label != 'feeling_response':
        randomNum = random.randint(0,50)
    user_input = input("You: ")
    if user_input == "exit":
        break
    else:
        if user_input == 'wrong':
            print(name+ last_label)
            new_label = input("Enter the correct label for the user input: ")
            if other_last_input in data['text'].values:
                continue
            else:
                new_row = pd.DataFrame({'text': [last_input], 'label': [new_label]})
                data = pd.concat([data, new_row], ignore_index=True)
                data.to_csv("conversational_english.csv", index=False)
        else:
            user_input_features = vectorizer.transform([user_input])
            predicted_label = nb.predict(user_input_features)[0]
            if not any(data['text'] == user_input):
                new_row = pd.DataFrame({'text': [user_input], 'label': [predicted_label]})
                data = pd.concat([data, new_row], ignore_index=True)
                data.to_csv("conversational_english.csv", index=False)
                train_and_save_model(data)
            if other_last_input not in data['text'].values and (other_last_input != '' and other_last_input != ' '):
                new_row = pd.DataFrame({'text': [other_last_input], 'label': [last_label]})
                data = pd.concat([data, new_row], ignore_index=True)
                data.to_csv("conversational_english.csv", index=False)
                train_and_save_model(data)
            if 'fav' in predicted_label and 'que' in predicted_label:
                last_label = predicted_label
                predicted_label = predicted_label.replace('que','response')      
                response = data.loc[data['label'] == predicted_label, 'text'].sample().values[0]
                print(name, response)
                last_input = user_input
            if predicted_label == 'feeling_question':
                last_label = predicted_label
                response = data.loc[data['label'] == 'feeling_response', 'text'].sample().values[0]
                print(name, response)
                if randomNum>=25:
                    print(name,data.loc[data['label'] == 'feeling_question', 'text'].sample().values[0])
                    new_user_input = input("You re: ")
                    if new_user_input == 'wrong':
                        print(name+ last_label)
                        new_label = input("Enter the correct label for the user input: ")
                        if user_input in data['text'].values:
                            continue
                        else:
                            new_row = pd.DataFrame({'text': [new_user_input], 'label': [new_label]})
                            data = pd.concat([data, new_row], ignore_index=True)
                            data.to_csv("conversational_english.csv", index=False)
                            train_and_save_model(data)
            if predicted_label == 'name_que':
                print(name + " My name is " + name.replace(':',''))
            if predicted_label == 'add_response':
                add = input('You:')
                answer = add.split(' ')
                add_answer = int(answer[0]) + int(answer[1])
                print(name, answer[0], '+', answer[1], '=', add_answer)
                continue
            other_last_input = last_input
            if predicted_label != 'feeling_question' or ('fav' in predicted_label and 'que' in predicted_label) or predicted_label != 'feeling_response' or predicted_label != 'joke_request':
                last_label = predicted_label
                response = data.loc[data['label'] == predicted_label, 'text'].sample().values[0]
                print(name, response)
            if last_label == 'farewell':
                endCheck = input("Would you like to end this session?")
                if endCheck == 'yes':
                    break
                else:
                    continue
                