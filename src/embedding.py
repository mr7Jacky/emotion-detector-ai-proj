import io
import numpy as np


def get_embeddings(file):
    """ This function reads the pre-trained embedding layers from file"""
    embeddings_index = {}
    dim = 0
    with io.open(file, encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            embedding_vector = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = embedding_vector
            dim = len(embedding_vector)
    return embeddings_index, dim


def get_embedding_matrix(embeddings, word_index, dim):
    """
    Calculate the embedding matrix from embeddings
    :param embeddings: the pretrained embedding layer
    :param word_index: all words
    :param dim: the dimension of words
    :return:
    """
    embedding_matrix = np.zeros((len(word_index) + 1, dim))
    for word, i in word_index.items():
        embedding_matrix[i] = embeddings.get(word)
    return embedding_matrix
