import numpy as np
import logging
from .config import ModelConfig as MC
from gensim.models import Word2Vec as w2v

logger = logging.getLogger(__name__)

def load_w2v():
    cod_mat = np.zeros((MC.CODE_SIZE, MC.COD_EMB_DIM))
    con_mat = np.zeros((MC.VOCAB_SIZE, MC.CON_EMB_DIM))
    tag_mat = np.zeros((MC.TAG_SIZE, MC.TAG_EMB_DIM))
    tit_mat = np.zeros((MC.VOCAB_SIZE, MC.TIT_EMB_DIM))
    cod_vec = w2v.load(MC.OUT_PATH+'/code.vec')
    con_vec = w2v.load(MC.OUT_PATH+'/content.vec')
    tag_vec = w2v.load(MC.OUT_PATH+'/tags.vec')
    tit_vec = w2v.load(MC.OUT_PATH+'/title.vec')
    for i in range(2,MC.CODE_SIZE-1):
        try:
            cod_mat[i] = cod_vec[str(i)]
        except:
            pass
    for i in range(2,MC.VOCAB_SIZE-1):
        try:
            con_mat[i] = con_vec[str(i)]
        except:
            pass
    for i in range(2,MC.TAG_SIZE-1):
        try:
            tag_mat[i] = tag_vec[str(i)]
        except:
            pass
    for i in range(2,MC.VOCAB_SIZE-1):
        try:
            tit_mat[i] = tit_vec[str(i)]
        except:
            pass
    return [cod_mat,con_mat,tag_mat,tit_mat]


def create_model():

    import keras
    import keras.backend as K
    from keras.models import Sequential, Model
    from keras.layers import concatenate, Embedding, Input, Flatten
    from nea.self_attention import Attention1D, Attention
    from nea.vat import Adversarial_Training
    from nea.layers import WordRepresLayer, CharRepresLayer, ContextLayer, PredictLayer, RNNLayer
    logger.info('Building model...')

    if MC.W2V:
        cod_mat,con_mat,tag_mat,tit_mat = load_w2v()

    con_in = Input(batch_shape=(None, MC.PAD_MAXLEN), dtype='int32')
    if MC.W2V:
        con_emb = Embedding(MC.VOCAB_SIZE, MC.CON_EMB_DIM, weights=[con_mat],trainable=False)(con_in)
    else:
        con_emb = Embedding(MC.VOCAB_SIZE, MC.CON_EMB_DIM)(con_in)
    if MC.CHAR_EMB:
        con_in2 = Input(batch_shape=(None, None, None), dtype='int32')
        char_layer = CharRepresLayer(
            MC.PAD_MAXLEN, MC.CHAR_SCALE, MC.MAX_CHAR_LEN, MC.CON_EMB_DIM,
            MC.CON_RNN_DIM, dropout=MC.DROPOUT)
        c_res1 = char_layer(con_in2)
        con_emb = concatenate([con_emb, c_res1])
    con_layer = RNNLayer(MC.RNN_UNIT, MC.CON_RNN_DIM, K.int_shape(con_emb)[
                         1:], MC.DROPOUT_W, MC.DROPOUT_U, MC.CNN_PADDING)(con_emb)

    cod_in = Input(batch_shape=(None, MC.PAD_MAXLEN), dtype='int32')
    if MC.W2V:
        cod_emb = Embedding(MC.CODE_SIZE, MC.COD_EMB_DIM, weights=[cod_mat],trainable=False)(cod_in)
    else:
        cod_emb = Embedding(MC.CODE_SIZE, MC.COD_EMB_DIM)(cod_in)
    if MC.CHAR_EMB:
        cod_in2 = Input(batch_shape=(None, None, None), dtype='int32')
        char_layer2 = CharRepresLayer(
            MC.PAD_MAXLEN, MC.CHAR_SCALE, MC.MAX_CHAR_LEN, MC.CON_EMB_DIM,
            MC.CON_RNN_DIM, dropout=MC.DROPOUT)
        c_res2 = char_layer2(cod_in2)
        cod_emb = concatenate([cod_emb, c_res2])
    cod_layer = RNNLayer(MC.RNN_UNIT, MC.COD_RNN_DIM, K.int_shape(cod_emb)[
                         1:], MC.DROPOUT_W, MC.DROPOUT_U, MC.CNN_PADDING)(cod_emb)

    tit_in = Input(batch_shape=(None, MC.PAD_MAXLEN), dtype='int32')
    if MC.W2V:
        tit_emb = Embedding(MC.VOCAB_SIZE, MC.TIT_EMB_DIM, weights=[tit_mat],trainable=False)(tit_in)
    else:
        tit_emb = Embedding(MC.VOCAB_SIZE, MC.TIT_EMB_DIM)(tit_in)
    if MC.CHAR_EMB:
        tit_in2 = Input(batch_shape=(None, None, None), dtype='int32')
        char_layer3 = CharRepresLayer(
            MC.PAD_MAXLEN, MC.CHAR_SCALE, MC.MAX_CHAR_LEN, MC.CON_EMB_DIM,
            MC.CON_RNN_DIM, dropout=MC.DROPOUT)
        c_res3 = char_layer3(tit_in2)
        tit_emb = concatenate([tit_emb, c_res3])
    tit_layer = RNNLayer(MC.RNN_UNIT, MC.TIT_RNN_DIM, K.int_shape(tit_emb)[
                         1:], MC.DROPOUT_W, MC.DROPOUT_U, MC.CNN_PADDING)(tit_emb)

    tag_in = Input(batch_shape=(None, None), dtype='int32')
    if MC.W2V:
        tag_layer = Embedding(MC.TAG_SIZE, MC.TAG_EMB_DIM, weights=[tag_mat],trainable=False)(tag_in)
    else:
        tag_layer = Embedding(MC.TAG_SIZE, MC.TAG_EMB_DIM)(tag_in)
    tag_layer = Attention()(tag_layer)

    output = concatenate([con_layer, cod_layer, tit_layer, tag_layer])
    output = PredictLayer(MC.DENSE_DIM, K.int_shape(
        output)[-1], MC.DROPOUT)(output)

    if MC.CHAR_EMB:
        model = Model(
            inputs=[con_in, con_in2, cod_in, cod_in2, tit_in, tit_in2, tag_in],
            outputs=[output]
        )
    else:
        model = Model(
            inputs=[con_in, cod_in, tit_in, tag_in],
            outputs=[output]
        )

    logger.info('  Done')
    model.summary()

    return model
