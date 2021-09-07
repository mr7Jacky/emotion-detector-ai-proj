# Ref: https://towardsdatascience.com/multi-class-text-classification-with-lstm-1590bee1bd17
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras import Model
from keras.layers import Dense, Bidirectional, Embedding, LSTM, SpatialDropout1D, Input, Concatenate, GaussianNoise
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Dropout, Flatten
import os
import io
import tensorflow as tf
import sys

"""
Import scripts
"""
# Append the source code directory in to file
sys.path.append("../src") 
from preprocessing import *
from embedding import *
from model_builder import *

"""
Define Constant
"""
# Max number of words in each dialogue.
MAX_SEQUENCE_LENGTH = 30 # Max number of vector length for each sentence
PLOT = False # Where to plot the learning curve
SIMPLE_CATEGORY = True # True to use 3 categories otherwise use 13 categories
CATEGORY_NUM = 5000 # number of training data for each emotions type (Max 9444) when using 3 categories

"""
Read data and preprocess it
"""
data = pd.read_csv('../Dataset/Tweets.csv')
text_col = 'content'
data = preprocess_text(data,text_col)
# Shuffle data
data = data.sample(frac=1).reset_index(drop=True)

"""
Convert into 3 categories
"""
if SIMPLE_CATEGORY:
    # Categorize into 3 categories
    data.loc[data['sentiment'] == 'anger'] = 'negative'
    data.loc[data['sentiment'] == 'hate'] = 'negative'
    data.loc[data['sentiment'] == 'worry'] = 'negative'
    data.loc[data['sentiment'] == 'sadness'] = 'negative'
    data.loc[data['sentiment'] == 'boredom'] = 'negative'
    data.loc[data['sentiment'] == 'relief'] = 'positive'
    data.loc[data['sentiment'] == 'happiness'] = 'positive'
    data.loc[data['sentiment'] == 'love'] = 'positive'
    data.loc[data['sentiment'] == 'enthusiasm'] = 'positive'
    data.loc[data['sentiment'] == 'surprise'] = 'positive'
    data.loc[data['sentiment'] == 'fun'] = 'positive'
    data.loc[data['sentiment'] == 'empty'] = 'neutral'
    # divide into 3 data to maintain a evenly distributed dataset
    neutral_data = data.loc[data['sentiment'] == 'neutral']
    negative_data = data.loc[data['sentiment'] == 'negative']
    positive_data = data.loc[data['sentiment'] == 'positive']
    # Obtain 5000 from each category
    data = pd.concat([neutral_data.sample(n=CATEGORY_NUM),
                      negative_data.sample(n=CATEGORY_NUM),
                      positive_data.sample(n=CATEGORY_NUM)])
    

"""
Create Embedding Layer
"""
embeddings, dim = get_embeddings("../glove/glove.6B.100d.txt")
tokenizer = get_tokenizer([' '.join(list(embeddings.keys()))])
embedding_matrix = get_embedding_matrix(embeddings, tokenizer.word_index, dim)

"""
Create training and testing sets
"""
X = tokenizer.texts_to_sequences(data['content'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
Y = pd.get_dummies(data['sentiment']).values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.50, random_state = 42)

"""
Train and evaluate the model
"""
model = build_model(embedding_matrix, MAX_SEQUENCE_LENGTH, lstm_dim=64, num_hid_layer=1, 
                    hidden_layer_dim=30, num_classes=13, noise=0.01, dropout_lstm=0.02, dropout=0.02)
epochs = 5
batch_size = 20
history = model.fit(X_train,Y_train, epochs=epochs, batch_size=batch_size,
                    validation_data=(X_test,Y_test),
                    #validation_split=0.5, 
                    verbose = 1,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.001)])
accr = model.evaluate(X_test,Y_test)
print(accr)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

"""
Plotting
"""
if PLOT:
    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show();
    plt.title('Accuracy')
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='test')
    plt.legend()
    plt.show();