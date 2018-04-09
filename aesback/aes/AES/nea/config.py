# -*- coding: utf-8 -*-
"""Model configs.
"""


class ModelConfig(object):

    ADDREGATION = 'attmean'  # mot|attsum|attmean
    BATCH_SIZE = 128
    CHAR_SCALE = 256
    CHAR_EMB = False
    CNN_PADDING = 'same'
    CNN_WINDOW_SIZE = 3
    COD_EMB_DIM = 300
    COD_RNN_DIM = 300
    CODE_SIZE = 200000
    CON_EMB_DIM = 300
    CON_RNN_DIM = 300
    CONVERT_SCORE = True
    CPU_ONLY = False
    DENSE_DIM = 300
    DEPTH = 2
    DEV_PATH = 'dataset/de.csv'
    DROPOUT = 0.5
    DROPOUT_U = 0.1
    DROPOUT_W = 0.5
    EMB_PATH = ''
    EPOCHS = 100
    HIGHWAY = False
    GPU = '0'
    LOSS_FUNC = 'kappa'  # mse|mae|kappa
    MAXLEN = 0  # Maximum allowed number of words during training. '0' means no limit
    MAX_CHAR_LEN = 50
    OPTIMIZER = 'amsgrad'  # rmsprop|sgd|adagrad|amsgrad|adadelta|adam|adamax
    OUT_PATH = 'output_dir'
    PAD_MAXLEN = 300
    PLOT_MODEL = False
    PRE_DATA = True
    RNN_UNIT = 'gru'  # lstm|gru|simple|sru|nlstm|indrnn
    SAVE_JSON = True
    SAVE_MODEL = True
    SEED = 5346
    TAG_EMB_DIM = 100
    TAG_SIZE = 20000
    TEST_PATH = 'dataset/te.csv'
    TIT_EMB_DIM = 100
    TIT_RNN_DIM = 300
    TRAIN_PATH = 'dataset/tr.csv'
    VAT = False
    VOCAB_PATH = 'aes/AES/output_dir/vocab.pkl'
    VOCAB_SIZE = 200000
    W2V = False
