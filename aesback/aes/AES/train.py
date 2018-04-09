#!/usr/bin/env python

import logging
import numpy as np
from time import time
import nea.utils as U
import pickle as pk
from keras.callbacks import TensorBoard
from nea.evaluator import Evaluator
import nea.reader as dataset
from nea.config import ModelConfig as MC
from sklearn.utils import class_weight

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

if MC.CPU_ONLY:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = MC.GPU

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
KTF.set_session(sess)


logger = logging.getLogger(__name__)

U.mkdir_p(MC.OUT_PATH + '/preds')
U.set_logger(MC.OUT_PATH)
np.random.seed(MC.SEED)

###############################################################################################################################
# Prepare data
#

# from keras.preprocessing import sequence
#
# # data_x is a list of lists
# (train_x, train_y), (dev_x, dev_y), (test_x, test_y), vocab, vocab_size, overal_maxlen, num_outputs = dataset.get_data(
#     MC.MAXLEN, to_lower=True, sort_by_len=False)
#
# # Dump vocab
# with open(MC.OUT_PATH + '/vocab.pkl', 'wb') as vocab_file:
#     pk.dump(vocab, vocab_file)
#
# if MC.CHAR_EMB:
#     for i in ['con_char', 'cod_char', 'tit_char']:
#         train_x[i] = np.array(train_x[i])
#         dev_x[i] = np.array(dev_x[i])
#         test_x[i] = np.array(test_x[i])

###############################################################################################################################
# Some statistics
#

# import keras.backend as K
#
# train_y = np.array(train_y, dtype=K.floatx())
# dev_y = np.array(dev_y, dtype=K.floatx())
# test_y = np.array(test_y, dtype=K.floatx())
#
# train_y_ori = train_y
# from keras.utils import to_categorical as toc
# train_y = toc(train_y,6)
# dev_y = toc(dev_y,6)
# test_y = toc(test_y,6)
#
# bincounts, mfs_list = U.bincounts(train_y)
# with open('%s/bincounts.txt' % MC.OUT_PATH, 'w') as output_file:
#     for bincount in bincounts:
#         output_file.write(str(bincount) + '\n')
#
# train_mean = train_y.mean(axis=0)
# train_std = train_y.std(axis=0)
# dev_mean = dev_y.mean(axis=0)
# dev_std = dev_y.std(axis=0)
# test_mean = test_y.mean(axis=0)
# test_std = test_y.std(axis=0)
#
# logger.info('Statistics:')
#
# for i in ['content', 'code', 'title', 'tags']:
#     logger.info('  train_' + i + ' shape: ' + str(np.array(train_x[i]).shape))
#     logger.info('  dev_' + i + ' shape:   ' + str(np.array(dev_x[i]).shape))
#     logger.info('  test_' + i + ' shape:  ' + str(np.array(test_x[i]).shape))
#
# logger.info('  train_y shape: ' + str(train_y.shape))
# logger.info('  dev_y shape:   ' + str(dev_y.shape))
# logger.info('  test_y shape:  ' + str(test_y.shape))
# logger.info('  train_y mean: %s, stdev: %s, MFC: %s' %
#             (str(train_mean), str(train_std), str(mfs_list)))
#
# # We need the dev and test sets in the original scale for evaluation
# dev_y_org = dev_y.astype(dataset.get_ref_dtype())
# test_y_org = test_y.astype(dataset.get_ref_dtype())
#
# ##############################################################################################################################
# Optimizaer algorithm
#
#
from nea.optimizers import get_optimizer
optimizer = get_optimizer()

###############################################################################################################################
# Building model
#

from nea.models import create_model
from keras import backend as K

def myloss(y_true,y_pred):
    return K.mean(K.square(K.square(y_true-y_pred)),axis=-1)

if MC.LOSS_FUNC == 'mse':
    loss = 'mean_squared_error'
    metric = 'mean_absolute_error'
elif MC.LOSS_FUNC == 'mae' :
    loss = 'mean_absolute_error'
    metric = 'mean_squared_error'
elif MC.LOSS_FUNC == 'kappa' :
    loss = 'categorical_crossentropy'
    metric = 'acc'
model = create_model()
model.compile(loss=loss, optimizer=optimizer, metrics=[metric])
model.load_weights('output_dir/final_2gru_20_20_20_cross.h5')
###############################################################################################################################
# Plotting model
#
if MC.PLOT_MODEL:
    from keras.utils import plot_model
    plot_model(model, to_file=MC.OUT_PATH + '/model.png', show_shapes=False)

###############################################################################################################################
# Save model architecture

if MC.SAVE_MODEL:
    logger.info('Saving model...')
    model.save('model.h5')
    logger.info('  Done')
if MC.SAVE_JSON:
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

###############################################################################################################################
# Evaluator
#

evl = Evaluator(dataset, MC.OUT_PATH, dev_x, test_x,
                dev_y, test_y, dev_y_org, test_y_org)

###############################################################################################################################
# Training
#

logger.info('--------------------------------------------------------------------------------------------------------------------------')
logger.info('Initial Evaluation:')
#evl.evaluate(model, -1, print_info=False)

total_train_time = 0
total_eval_time = 0

c_weight = class_weight.compute_class_weight('balanced',np.array([0,1,2,3,4,5]),train_y_ori)
print(c_weight)
c_dic = {}
for i in range(6):
    c_dic[i] = c_weight[i]

for ii in range(MC.EPOCHS):
    t0 = time()
    if MC.CHAR_EMB:
        train_history = model.fit(
            [
                train_x['content'], train_x['con_char'], train_x['code'], train_x['cod_char'], train_x['title'], train_x['tit_char'], train_x['tags']],
            train_y,
            batch_size=MC.BATCH_SIZE,
            epochs=1,
            verbose=0,
            callbacks=[TensorBoard(write_graph=True, write_grads=True, write_images=True, log_dir='./temp/log')
                       ])
    else:
        train_history = model.fit(
            [
                train_x['content'], train_x['code'], train_x['title'], train_x['tags']],
            train_y,
            batch_size=MC.BATCH_SIZE,
            epochs=1,
            verbose=1,
            class_weight=c_dic
            )
    tr_time = time() - t0
    total_train_time += tr_time

    # Evaluate
    t0 = time()
    evl.evaluate(model, ii,False)
    evl_time = time() - t0
    total_eval_time += evl_time

    # Print information
    train_loss = train_history.history['loss'][0]
    train_metric = train_history.history[metric][0]
    logger.info('Epoch %d, train: %is, evaluation: %is' %
                (ii, tr_time, evl_time))
    logger.info('[Train] loss: %.4f, metric: %.4f' %
                (train_loss, train_metric))
#    evl.print_info()

###############################################################################################################################
# Summary of the results
#

    model.save_weights(MC.OUT_PATH+'/final_3ind_20_20_20_cross.h5',overwrite=True)

logger.info('Training:   %i seconds in total' % total_train_time)
logger.info('Evaluation: %i seconds in total' % total_eval_time)
