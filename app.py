import streamlit as st
import pandas as pd
import nltk
import string

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# NLTK download
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

# Text preprocessing
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []

    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load dataset
df = pd.read_csv('spam.csv', encoding='latin1')

df = df[['v1', 'v2']]
df.columns = ['target', 'text']

# Encode target
df['target'] = df['target'].map({'ham':0, 'spam':1})

# Transform text
df['transformed_text'] = df['text'].apply(transform_text)

# TF-IDF
tfidf = TfidfVectorizer(max_features=3000)

X = tfidf.fit_transform(df['transformed_text'])

y = df['target']

# Train model
model = MultinomialNB()

model.fit(X, y)

# Streamlit UI
st.title("SMS Spam Classifier")

input_sms = st.text_area("Enter Message")

if st.button("Predict"):

    transformed_sms = transform_text(input_sms)

    vector_input = tfidf.transform([transformed_sms])

    result = model.predict(vector_input)[0]

    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
