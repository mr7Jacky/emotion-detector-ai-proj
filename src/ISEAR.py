import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.preprocessing.text import Tokenizer
import os
import io
import tensorflow as tf
import sys
import nltk

# Append the source code directory in to file
sys.path.append("../src") 
from preprocessing import *
from embedding import *
from model_builder import *

"""
Define Constant
"""
# Max number of words in each dialogue.
MAX_SEQUENCE_LENGTH = 160 # Max number of vector length for each sentence
PLOT = False # Where to plot the learning curve

"""
Obtain the dataset and preprocessing the data
"""
# Read the data
data = pd.read_csv("../Dataset/ISEAR/isear_databank.csv")
# Preprocess the text data
text_col = 'SIT'
data = preprocess_text(data,text_col)

"""
Create the embedding layer
"""
# create embedding layers from file
embeddings, dim = get_embeddings("../glove/glove.6B.100d.txt")
# Fit the words in embedding layers to the tokenizer
tokenizer = get_tokenizer([' '.join(list(embeddings.keys()))])
# Get embedding matrix
embedding_matrix = get_embedding_matrix(embeddings, tokenizer.word_index, dim)


"""
We obtain the training and testing data from column storing text info and sentiments info into X and Y, 
where X is the input sentences, Y is expected output label.
"""
# Vectorize the text into number sequence
X = tokenizer.texts_to_sequences(data['SIT'].values)
# Add length to each vector in X to reach a same length
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
# Create a one-hot representation for each label in dataset and stored in Y
Y = pd.get_dummies(data[ 'Field1']).values
# Splot the data into two subset for training and testing
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.050, random_state = 42)

"""
Model creation
"""
# Build the model
model = build_model(embedding_matrix, MAX_SEQUENCE_LENGTH, lstm_dim=64, num_hid_layer=1, 
                    hidden_layer_dim=30, num_classes=7)

"""
Training and Evaluate
"""
# train the model
epochs = 30
batch_size = 30
history = model.fit(X_train,Y_train, epochs=epochs, batch_size=batch_size, 
                    validation_data=(X_test, Y_test), verbose = 1,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=4, min_delta=0.005, restore_best_weights=True)])

# evaluate accuracy
accr = model.evaluate(X_test,Y_test)
print(accr)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

"""
Plotting
"""
if PLOT:
    # Plotting the loss 
    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show();
    # Plotting the accuracy
    plt.title('Accuracy')
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='test')
    plt.legend()
    plt.show();