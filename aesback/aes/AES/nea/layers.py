from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Lambda, Dense, Dropout, Activation
from keras.layers.recurrent import LSTM, GRU
from keras.layers.wrappers import Bidirectional
from keras.legacy.layers import Highway
from keras.layers import TimeDistributed, concatenate, Embedding, Conv1D, Lambda
import keras.backend as K
from keras.layers.normalization import BatchNormalization
from .self_attention import Attention1D, Attention
from .config import ModelConfig as MC
import numpy as np


class WordRepresLayer(object):
    """Word embedding representation layer
    """

    def __init__(self, nb_words, word_embedding_dim):
        self.model = Sequential()
        self.model.add(
            Embedding(nb_words, word_embedding_dim, trainable=False))

    def __call__(self, inputs):
        return self.model(inputs)


class CharRepresLayer(object):
    """Char embedding representation layer
    """

    def __init__(self, sequence_length, nb_chars, nb_per_word,
                 embedding_dim, rnn_dim, rnn_unit='gru', dropout=0.0):
        def _collapse_input(x, nb_per_word=0):
            x = K.reshape(x, (-1, nb_per_word))
            return x

        def _unroll_input(x, sequence_length=0, rnn_dim=0):
            x = K.reshape(x, (-1, sequence_length, rnn_dim))
            return x

        if rnn_unit == 'gru':
            rnn = GRU
        else:
            rnn = LSTM
        self.model = Sequential()
        self.model.add(Lambda(_collapse_input,
                              arguments={'nb_per_word': nb_per_word},
                              output_shape=(nb_per_word,),
                              input_shape=(sequence_length, nb_per_word,)))
        self.model.add(Embedding(nb_chars,
                                 embedding_dim,
                                 input_length=nb_per_word,
                                 trainable=True))
        self.model.add(rnn(rnn_dim,
                           dropout=dropout,
                           recurrent_dropout=dropout))
        self.model.add(Lambda(_unroll_input,
                              arguments={'sequence_length': sequence_length,
                                         'rnn_dim': rnn_dim},
                              output_shape=(sequence_length, rnn_dim)))

    def __call__(self, inputs):
        return self.model(inputs)


class ContextLayer(object):
    """Word context layer
    """

    def __init__(self, rnn_dim, rnn_unit='gru', input_shape=(0,),
                 dropout=0.0, highway=False, return_sequences=False,
                 dense_dim=0):
        if rnn_unit == 'gru':
            rnn = GRU
        else:
            rnn = LSTM
        self.model = Sequential()
        self.model.add(
            Bidirectional(rnn(rnn_dim,
                              dropout=dropout,
                              recurrent_dropout=dropout,
                              return_sequences=return_sequences),
                          input_shape=input_shape))
        if highway:
            if return_sequences:
                self.model.add(TimeDistributed(Highway(activation='tanh')))
            else:
                self.model.add(Highway(activation='tanh'))

        if dense_dim > 0:
            self.model.add(TimeDistributed(Dense(dense_dim,
                                                 activation='relu')))
            self.model.add(TimeDistributed(Dropout(dropout)))
            self.model.add(TimeDistributed(BatchNormalization()))

    def __call__(self, inputs):
        return self.model(inputs)


class RNNLayer(object):
    """RNN layer.
    """

    def __init__(self, rnn, rnn_dim, input_dim, dropout_W=0.0, dropout_U=0.0, cnn_border_mode='same'):
        if rnn == 'lstm':
            from keras.layers import CuDNNLSTM as RNN
        elif rnn == 'sru':
            from nea.cell import SRU as RNN
        elif rnn == 'nlstm':
            from nea.cell import NestedLSTM as RNN
        elif rnn == 'gru':
            from keras.layers import CuDNNGRU as RNN
        elif rnn == 'simple':
            from keras.layers.recurrent import SimpleRNN as RNN
        elif rnn == 'indrnn':
            from nea.cell import IndRNN as RNN
        self.model = Sequential()
        self.model.add(Conv1D(filters=100, kernel_size=3,
                              padding=cnn_border_mode, strides=1, input_shape=input_dim))
        for i in range(MC.DEPTH):
            self.model.add(
                Bidirectional(RNN(rnn_dim,
                                  # dropout=dropout_W,
                                  #recurrent_dropout=dropout_U,
                                  return_sequences=True),
                              ))

        if MC.HIGHWAY:
            self.model.add(TimeDistributed(Highway(activation='tanh')))
        #self.model.add(TimeDistributed(Dense(MC.DENSE_DIM,activation='relu')))
        self.model.add(Dropout(MC.DROPOUT))
        self.model.add(Attention())
        #self.model.add(Dropout(MC.DROPOUT))
        # self.model.add(TimeDistributed(Dropout(MC.DROPOUT)))


    def __call__(self, inputs):
        return self.model(inputs)

def out_shape(input):
    return (input[0],1)

class PredictLayer(object):
    """Prediction layer.
    """

    def __init__(self, dense_dim, input_dim=0,
                 dropout=0.0):
        self.model = Sequential()
        self.model.add(Dense(dense_dim,
                             activation='tanh',
                             input_shape=(input_dim,)))
        self.model.add(Dropout(dropout))
        self.model.add(Dense(6, activation='softmax'))
        #self.model.add(Lambda(lambda x:K.cast(K.dot(x,K.variable(np.array([[0],[1],[2],[3],[4],[5]]))),'float32'),output_shape=out_shape))

    def __call__(self, inputs):
        return self.model(inputs)
