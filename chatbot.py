import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from textblob import TextBlob
from difflib import get_close_matches
import openai

# Initialize the OpenAI API
openai.api_key = 'Acche se apni key daalna :)-'

# Load data


lemmatizer = WordNetLemmatizer()

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)

    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence, words)
    res = model.predict(np.array([bow]))[0]
    error_threshold = 0.25
    result = [[i, r] for i, r in enumerate(res) if r > error_threshold]
    result.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in result:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    if not intents_list:
        return "Saathi: I'm not sure how to respond to that."

    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    else:
        result = "Saathi: I'm not sure how to respond to that."

    return "Saathi: " + result.split("|")[0].strip()

def find_best_matches(user_question: str, questions: list[str]) -> str | None:
    matches: list = get_close_matches(user_question, questions, n=1, cutoff=0.6)
    return matches[0] if matches else None

def get_random_answer(question_answers):
    return random.choice(question_answers)

def load_knowledge_base(file_path: str) -> dict:
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def chatbot_response(msg):
    blob = TextBlob(msg)
    sentiment_polarity = blob.sentiment.polarity
    
    if sentiment_polarity == 0:
        ints = predict_class(msg)
        res = get_response(ints, intents)
        if "I'm not sure how to respond to that" not in res:
            return res
    else:
        best_match = find_best_matches(msg, [q['question'] for q in knowledge_base['questions']])
        
        if best_match:
            question_data = next(q for q in knowledge_base['questions'] if q['question'] == best_match)
            ans = get_random_answer(question_data['answers'])
            res = "Saathi: " + ans.split("|")[0].strip()
            return res
    
    # If question not found in knowledge base or intents, use OpenAI for the response
    messages = [{"role": "system", "content": "You are an intelligent assistant."}, {"role": "user", "content": msg}]
    
    # Create a chat completion using GPT-3 with max tokens to limit response length
    chat = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.5,
        max_tokens=70,  # Adjust this value to control the response length
    )

    # Get the assistant's reply from the completion
    assistant_reply = chat.choices[0].message.content

    res = "Saathi: " + assistant_reply.split("|")[0].strip()
    return res

intents = json.loads(open('intents.json').read())
knowledge_base = load_knowledge_base('knowledge_base.json')
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')
print("Bot is running:")

