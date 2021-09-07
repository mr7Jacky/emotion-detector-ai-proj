from keras.models import Sequential
from keras.layers import Dense, Bidirectional, LSTM, SpatialDropout1D, GaussianNoise, Dropout, Embedding
import tensorflow as tf


def build_model(embeddings_matrix, sequence_length, lstm_dim, num_hid_layer, hidden_layer_dim, num_classes,
                noise=0.1, dropout_lstm=0.2, dropout=0.2):
    """
    Create te Bidirectional LSTM model with an embedding input layer
    :param embeddings_matrix: the matrix contains embedding information of input sentences
    :param sequence_length: the length of input sentence
    :param lstm_dim: the number of node in lstm
    :param num_hid_layer: number of hidden layer
    :param hidden_layer_dim: the number of node in hidden layer
    :param num_classes: the number of class to classify
    :param noise: percentage of noise
    :param dropout_lstm: percentage of dropout in lstm
    :param dropout: percentage of dropout
    :return:
    """
    embedding_layer, embedding_dim = create_embedding(embeddings_matrix)
    model = Sequential()
    model.add(embedding_layer)
    model.add(GaussianNoise(noise, input_shape=(None, sequence_length, embedding_dim)))
    model.add(SpatialDropout1D(dropout))
    model.add(Bidirectional(LSTM(lstm_dim, dropout=dropout_lstm)))
    model.add(Dropout(dropout))
    for _ in range(num_hid_layer):
        model.add(Dense(hidden_layer_dim, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


def create_embedding(embeddings_matrix, max_sequence_length=100):
    """
    Create am embedding layer
    :param embeddings_matrix: the matrix contains embedding information of input sentences
    :param max_sequence_length: the length of input sentence
    :return:
    """
    embedding_dim = embeddings_matrix.shape[1]
    embedding_layer = Embedding(embeddings_matrix.shape[0],
                                embedding_dim,
                                weights=[embeddings_matrix],
                                input_length=max_sequence_length,
                                trainable=False)
    return embedding_layer, embedding_dim
