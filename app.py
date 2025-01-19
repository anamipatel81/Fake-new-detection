from flask import Flask, request, render_template
from pyngrok import ngrok
import pickle
import re
import numpy as np
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from textstat import textstat
import nltk

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')  
nltk.download('wordnet')
nltk.download('vader_lexicon')

with open('tfidf_title.pkl', 'rb') as f:
    tfidf_title = pickle.load(f)

with open('tfidf_text.pkl', 'rb') as f:
    tfidf_text = pickle.load(f)

with open('svd_title.pkl', 'rb') as f:
    svd_title = pickle.load(f)

with open('svd_text.pkl', 'rb') as f:
    svd_text = pickle.load(f)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)  
    text = text.lower()
    words = word_tokenize(text)  
    words = [word for word in words if word not in stopwords.words('english')]
    pos_tags = pos_tag(words)
    lemmatized_words = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]
    return ' '.join(lemmatized_words)

def feature_engineering(title, text):
    title_length = len(title.split())
    text_length = len(text.split())
    title_readability_score = textstat.flesch_kincaid_grade(title)
    text_readability_score = textstat.flesch_kincaid_grade(text)
    sia = SentimentIntensityAnalyzer()
    title_sentiment = sia.polarity_scores(title)['compound']
    text_sentiment = sia.polarity_scores(text)['compound']
    title_word_density = title_length / (text_length + 1) 
    
    return [title_length, text_length, title_readability_score, text_readability_score, 
            title_sentiment, text_sentiment, title_word_density]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    title = request.form['title']
    text = request.form['text']

    cleaned_title = clean_text(title)
    cleaned_text = clean_text(text)

    title_features = tfidf_title.transform([cleaned_title])
    text_features = tfidf_text.transform([cleaned_text])   

    title_features = svd_title.transform(title_features)
    text_features = svd_text.transform(text_features)

    additional_features = feature_engineering(title, text)

    final_features = np.hstack((title_features, text_features, [additional_features]))

    prediction = model.predict(final_features)
    probability = model.predict_proba(final_features)[:, 1]

    result = "Real" if prediction[0] == 1 else "Fake"
    return render_template('index.html', prediction=result)


if __name__ == '__main__':
    app.run(debug=True)
