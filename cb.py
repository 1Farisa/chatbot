
import streamlit as st
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import random

import json


with open(r'C:\Users\AcerAspireE15\Downloads\intents.json', 'r') as f:
    data = json.load(f)



df = pd.read_json(r'C:\Users\AcerAspireE15\Downloads\intents.json')

new_intents_data = {
    "tag": ["symptom"] * 4,  
    "patterns": [
        "I have a headache",
        "I feel dizzy",
        "I'm not feeling well",
        "I have a cough"
    ],
    "responses": [
        "I'm sorry to hear that you're not feeling well. It's best to consult with a healthcare professional.",
        "Headaches can be caused by various factors. Make sure to stay hydrated and consider resting.",
        "Dizziness can occur due to various reasons; please consult a doctor if it persists.",
        "Coughing can be a sign of various conditions. If it continues, please seek medical advice."
    ]
}


new_intents_df = pd.DataFrame(new_intents_data)


df = pd.concat([df, new_intents_df], ignore_index=True)



dic = {"tag": [], "patterns": [], "responses": []}

for intent in data['intents']:
    tag = intent['tag']
    patterns = intent['patterns']
    responses = intent['responses']
    for pattern in patterns:
        dic['tag'].append(tag)
        dic['patterns'].append(pattern)
        dic['responses'].append(responses)  
df = pd.DataFrame.from_dict(dic)


X = df['patterns']
y = df['tag']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = SVC()
model.fit(X_train_vec, y_train)


def predict_intent(user_input):
  
    user_input_vec = vectorizer.transform([user_input])
   
    intent = model.predict(user_input_vec)[0]
    return intent

def generate_response(intent):
   
    possible_responses = df[df['tag'] == intent]['responses'].values[0]
    
    response = random.choice(possible_responses)
    return response


st.title("Chatbot")
st.write("chatgpt")
st.write("chatbot for mental health conversation .")

user_input = st.text_input("You:", "")


if user_input:
    
    intent = predict_intent(user_input)

    response = generate_response(intent)
    st.write("Chatbot:", response)
