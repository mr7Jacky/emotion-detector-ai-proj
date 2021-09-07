""" This scripts provides functions to preprocessing text data for natural language processing"""
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
import re

def get_tokenizer(words):
    """
    Get the tokenizer of given input words
    input words should contains all the words in dataset
    :param words: string if all words
    :return: Tokenizer object fot converting words to tokens
    """
    tokenizer = Tokenizer(filters='')
    tokenizer.fit_on_texts(words)
    return tokenizer


def preprocess_text(df: pd.DataFrame, text_col: str):
    """
    Preprocess the text content inside a pandas dataframe.
    This function will add a new column for preprocessed data named "CleanMsg"
    :param df: the input dataframe object
    :param text_col: the column name containing text in dataframe
    :return: Dataframe
    """
    # drop nan rows
    df = df.dropna(subset=[text_col])
    # Apply cleaning function to text
    df['CleanMsg'] = df[text_col].apply(clean_url)
    df['CleanMsg'] = df['CleanMsg'].apply(clean_non_alphanumeric)
    df['CleanMsg'] = df['CleanMsg'].apply(clean_lowercase)
    df['CleanMsg'] = df['CleanMsg'].apply(clean_tokenization)
    df['CleanMsg'] = df['CleanMsg'].apply(clean_stopwords)
    df['CleanMsg'] = df['CleanMsg'].apply(clean_lemma)
    df['CleanMsg'] = df['CleanMsg'].apply(clean_length2)
    # Cleaning rows with 0 length text
    df = df[df['CleanMsg'].str.len() != 0]
    df = df.reset_index()
    return df


def clean_url(text):
    """
    Cleaning the url from text
    :param text: input text
    :return: text without URL
    """
    return re.sub(r'http\S+', '', text)


def clean_non_alphanumeric(text):
    """
    Remove non alphabetical characters in text
    :param text: input text
    :return: text contains only alphabets
    """
    return re.sub('[^a-zA-Z]', ' ', text)


def clean_lowercase(text):
    """
    Lower all the character in text
    :param text: input text
    :return: text with lowered cases
    """
    return str(text.lower())


def clean_tokenization(text):
    """
    Tokenize the word
    :param text: input text
    :return: list of all word in current text
    """
    return word_tokenize(text)


def clean_stopwords(token):
    """
    Remove the stop words from the tokens
    :param token: a list of words as tokens
    :return: the rest of list after cleaning
    """
    stopwords_eng = stopwords.words('english')
    return [item for item in token if item not in stopwords_eng]


def clean_lemma(token):
    """
    Remove all the tense in word
    :param token: a list of words as tokens
    :return: the rest of list after cleaning
    """
    lemma = WordNetLemmatizer()
    return [lemma.lemmatize(word=w, pos='v') for w in token]


def clean_length2(token):
    """
    Remove all the word with length less than 2
    :param token: a list of words as tokens
    :return: the rest of list after cleaning
    """
    return [i for i in token if len(i) > 2]
