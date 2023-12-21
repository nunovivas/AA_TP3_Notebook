import pandas as pd
import spacy
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer

def preprocess_text(text):
      # remove html tags
    text = text.replace('<br />', ' ')

    # remove punctuation and '_'
    for char in ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']:
        text = text.replace(char, ' ')

    # convert to lower case
    text = text.lower()

    # remove stop words
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])

    # remove numbers
    text = ''.join([char for char in text if not char.isdigit()])

    # remove extra spaces
    text = ' '.join(text.split())

    # replace words with their root form
    stemmer = SnowballStemmer('english')
    text = ' '.join([stemmer.stem(word) for word in text.split()])

    # replace words with their lemma
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

    # Remove people's names using spaCy
    nlp = spacy.load("en_core_web_sm")
    text = ' '.join([token.text if token.ent_type_ != 'PERSON' else 'NamePlaceholder' for token in nlp(text)])

    return text