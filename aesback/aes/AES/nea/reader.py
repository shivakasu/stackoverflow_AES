import random
import codecs
import sys
import nltk
import logging
import re
import numpy as np
import pickle as pk
import pandas as pd
from .config import ModelConfig as MC
from nltk.stem import WordNetLemmatizer


logger = logging.getLogger(__name__)
num_regex = re.compile('^[+-]?[0-9]+\.?[0-9]*$')
ref_scores_dtype = 'int32'

score_ranges = (-100, 1000)
aver = 0.0
vara = 0.0


def get_ref_dtype():
    return ref_scores_dtype

def str_expand(t):
    pat_is = re.compile("(it|he|she|that|this|there|here)(\'s)", re.I)
    pat_not = re.compile("(?<=[a-zA-Z])n\'t")
    pat_would = re.compile("(?<=[a-zA-Z])\'d")
    pat_will = re.compile("(?<=[a-zA-Z])\'ll")
    pat_am = re.compile("(?<=[I|i])\'m")
    pat_are = re.compile("(?<=[a-zA-Z])\'re")
    pat_ve = re.compile("(?<=[a-zA-Z])\'ve")
    s = pat_is.sub(r"\1 is", t)
    s = pat_not.sub(" not", s)
    s = pat_would.sub(" would", s)
    s = pat_will.sub(" will", s)
    s = pat_am.sub(" am", s)
    s = pat_are.sub(" are", s)
    s = pat_ve.sub(" have", s)
    return s

def tokenize(string):
    lemmatizer=WordNetLemmatizer()
    tokens = str_expand(string)
    tokens = nltk.word_tokenize(tokens)
    for i in range(len(tokens)):
        w = tokens[i]
        w1 = lemmatizer.lemmatize(w,pos='n')
        if w1==w:
            w1 = lemmatizer.lemmatize(w,pos='a')
        if w1==w:
            w1 = lemmatizer.lemmatize(w,pos='v')
        tokens[i] = w1
    for index, token in enumerate(tokens):
        if token == '@' and (index + 1) < len(tokens):
            tokens[index + 1] = '@' + re.sub('[0-9]+.*', '', tokens[index + 1])
            tokens.pop(index)
    return tokens


def get_score_range():
    return score_ranges


def get_model_friendly_scores(scores_array):
    if MC.CONVERT_SCORE:
        scores_array = (scores_array-aver)/vara
    return scores_array


def convert_to_dataset_friendly_scores(scores_array):
    if MC.CONVERT_SCORE:
        scores_array = scores_array * vara + aver
    return scores_array


def score_to_rank(s):
    if s<-3:
        return 0
    elif (s>=-3 and s<0):
        return 1
    elif (s>=0 and s<4):
        return 2
    elif (s>=4 and s<10):
        return 3
    elif (s>=10 and s<100):
        return 4
    else:
        return 5


def is_number(token):
    return bool(num_regex.match(token))


def load_vocab(file_path):
    logger.info('Loading from: ' + file_path)
    with open(file_path, 'rb') as vocab_file:
        vocab = pk.load(vocab_file)
    return vocab

def code_split(s):
    res = []
    start = 0
    for i in range(len(s)):
        if not (s[i].isalpha() or s[i].isdigit()):
            if start < i:
                res.append(s[start:i])
            if s[i:i + 1] != ' ':
                res.append(s[i:i + 1])
            start = i + 1
    return res


def create_vocab(maxlen, to_lower):
    logger.info('Creating vocabulary from: ' + MC.TRAIN_PATH)
    if maxlen > 0:
        logger.info('  Removing sequences with more than ' +
                    str(maxlen) + ' words')
    total_content, unique_content = 0, 0
    total_code, unique_code = 0, 0
    total_tags, unique_tags = 0, 0
    content_freqs = {}
    code_freqs = {}
    tags_freqs = {}

    def read_content(s):
        nonlocal unique_content,total_content
        content = str(s)
        if to_lower:
            content = content.lower()
        content = tokenize(content)
        for word in content:
            if is_number(word):
                continue
            try:
                content_freqs[word] += 1
            except KeyError:
                unique_content += 1
                content_freqs[word] = 1
            total_content += 1

    def read_title(s):
        nonlocal unique_content,total_content
        title = str(s)
        if to_lower:
            title = title.lower()
        title = tokenize(title)
        for word in title:
            if is_number(word):
                continue
            try:
                content_freqs[word] += 1
            except KeyError:
                unique_content += 1
                content_freqs[word] = 1
            total_content += 1

    def read_code(s):
        nonlocal unique_code,total_code
        code = str(s)
        if to_lower:
            code = code.lower()
        code = code_split(code)
        for word in code:
            if is_number(word):
                continue
            try:
                code_freqs[word] += 1
            except KeyError:
                unique_code += 1
                code_freqs[word] = 1
            total_code += 1

    def read_tags(s):
        nonlocal unique_tags,total_tags
        tags = str(s)
        if to_lower:
            tags = tags.lower()
        tags = re.findall('<(.+?)>', tags)
        for word in tags:
            try:
                tags_freqs[word] += 1
            except KeyError:
                unique_tags += 1
                tags_freqs[word] = 1
            total_tags += 1

    df = pd.read_csv(MC.TRAIN_PATH, encoding='latin1')
    df.Body.apply(lambda s:read_content(s))
    df.Code.apply(lambda s:read_code(s))
    df.Tags.apply(lambda s:read_tags(s))
    df.Title.apply(lambda s:read_title(s))
    logger.info('content  %i total words, %i unique words' %
                (total_content, unique_content))
    logger.info('code     %i total words, %i unique words' %
                (total_code, unique_code))
    logger.info('tags     %i total words, %i unique words' %
                (total_tags, unique_tags))
    del(df)
    import operator
    sorted_content_freqs = sorted(
        content_freqs.items(), key=operator.itemgetter(1), reverse=True)
    sorted_code_freqs = sorted(
        code_freqs.items(), key=operator.itemgetter(1), reverse=True)
    sorted_tags_freqs = sorted(
        tags_freqs.items(), key=operator.itemgetter(1), reverse=True)
    content_vocab = {'<pad>': 0, '<unk>': 1, '<num>': 2}
    code_vocab = {'<pad>': 0, '<unk>': 1, '<num>': 2}
    tags_vocab = {'<pad>': 0, '<unk>': 1, '<num>': 2}
    index = 3
    for word, _ in sorted_content_freqs:
        content_vocab[word] = index
        index += 1
    index = 3
    for word, _ in sorted_code_freqs:
        code_vocab[word] = index
        index += 1
    index = 3
    for word, _ in sorted_tags_freqs:
        tags_vocab[word] = index
        index += 1
    vocab = {
        'content': content_vocab,
        'code': code_vocab,
        'tags': tags_vocab
    }
    return vocab


def vectorize(vocab, wordlist):
    indices = []
    for word in wordlist:
        if is_number(word):
            indices.append(vocab['<num>'])
        elif word in vocab:
            indices.append(vocab[word])
        else:
            indices.append(vocab['<unk>'])
    return indices


def get_char_seq(words):
    seq = [[0 for i in range(MC.MAX_CHAR_LEN)] for j in range(MC.PAD_MAXLEN)]
    for k in range(len(words)):
        word = list(words[k])
        for m in range(len(word)):
            if m > (MC.MAX_CHAR_LEN - 1) or k > (MC.PAD_MAXLEN - 1):
                break
            seq[k][m] = ord(word[m])
    return seq


def read_dataset(file_path, maxlen, vocab, to_lower):
    logger.info('Reading dataset from: ' + file_path)
    if maxlen > 0:
        logger.info('  Removing sequences with more than ' +
                    str(maxlen) + ' words')
    if MC.CHAR_EMB:
        data_x = {'content': [], 'code': [], 'title': [], 'tags': [],
                  'con_char': [], 'cod_char': [], 'tit_char': []}
    else:
        data_x = {'content': [], 'code': [], 'title': [], 'tags': []}
    data_y = []
    maxlen_x = {'content': 0, 'code': 0, 'title': 0, 'tags': 0}

    def read_content(s):
        content = str(s)
        if to_lower:
            content = content.lower()
        content = tokenize(content)
        if MC.CHAR_EMB:
            data_x['con_char'].append(get_char_seq(content))
        indices = vectorize(vocab['content'], content)
        data_x['content'].append(indices)
        if maxlen_x['content'] < len(indices):
            maxlen_x['content'] = len(indices)

    def read_code(s):
        code = str(s)
        if to_lower:
            code = code.lower()
        code = code_split(code)
        if MC.CHAR_EMB:
            data_x['cod_char'].append(get_char_seq(code))
        indices = vectorize(vocab['code'], code)
        data_x['code'].append(indices)
        if maxlen_x['code'] < len(indices):
            maxlen_x['code'] = len(indices)

    def read_title(s):
        title = str(s)
        if to_lower:
            title = title.lower()
        title = tokenize(title)
        if MC.CHAR_EMB:
            data_x['tit_char'].append(get_char_seq(title))
        indices = vectorize(vocab['content'], title)
        data_x['title'].append(indices)
        if maxlen_x['title'] < len(indices):
            maxlen_x['title'] = len(indices)

    def read_tags(s):
        tags = str(s)
        if to_lower:
            tags = tags.lower()
        tags = re.findall('<(.+?)>', tags)
        indices = vectorize(vocab['tags'], tags)
        data_x['tags'].append(indices)
        if maxlen_x['tags'] < len(indices):
            maxlen_x['tags'] = len(indices)

    def read_score(s):
        score = float(s)
        data_y.append(score)

    df = pd.read_csv(file_path, encoding='latin1')
    df.Body.apply(lambda s:read_content(s))
    df.Code.apply(lambda s:read_code(s))
    df.Tags.apply(lambda s:read_tags(s))
    df.Title.apply(lambda s:read_title(s))
    df.Score.apply(lambda s:read_score(s))

    for i in ['content','code','tags','title']:
        data_x[i] = np.array(data_x[i])

    return data_x, data_y, maxlen_x

def get_data(maxlen, to_lower=True, sort_by_len=False):

    global aver,vara

    if not MC.VOCAB_PATH:
        vocab = create_vocab(maxlen, to_lower)
        if len(vocab['content']) < MC.VOCAB_SIZE:
            logger.warning('The vocabualry includes only %i words (less than %i)' % (
                len(vocab['content']), MC.VOCAB_SIZE))
    else:
        vocab = load_vocab(MC.VOCAB_PATH)
    logger.info('  Vocab total: %i' % (len(vocab['content'])))
    logger.info('  code  total: %i' % (len(vocab['code'])))
    logger.info('  tags  total: %i' % (len(vocab['tags'])))
    logger.info('  Vocab use  : %i' % (MC.VOCAB_SIZE))
    logger.info('  code  use  : %i' % (MC.CODE_SIZE))
    logger.info('  tags  use  : %i' % (MC.TAG_SIZE))

    if not MC.PRE_DATA:
        train_x, train_y,  train_maxlen = read_dataset(MC.TRAIN_PATH, maxlen, vocab, to_lower)
        dev_x, dev_y, dev_maxlen = read_dataset(MC.DEV_PATH, 0, vocab, to_lower)
        test_x, test_y, test_maxlen = read_dataset(MC.TEST_PATH, 0, vocab, to_lower)
        with open(MC.OUT_PATH + '/train.pkl', 'wb') as data_file:
            pk.dump([train_x, train_y, train_maxlen], data_file)
        with open(MC.OUT_PATH + '/dev.pkl', 'wb') as data_file:
            pk.dump([dev_x, dev_y, dev_maxlen], data_file)
        with open(MC.OUT_PATH + '/test.pkl', 'wb') as data_file:
            pk.dump([test_x, test_y, test_maxlen], data_file)

    else:
        train_x, train_y,  train_maxlen = load_vocab(MC.OUT_PATH+"/train.pkl")
        dev_x, dev_y, dev_maxlen = load_vocab(MC.OUT_PATH+"/dev.pkl")
        test_x, test_y, test_maxlen = load_vocab(MC.OUT_PATH+"/test.pkl")

    aver = np.average(train_y)
    vara = np.std(train_y)

    from keras.preprocessing import sequence

    for i in ['content','code','tags','title']:
        train_x[i] = sequence.pad_sequences(train_x[i], MC.PAD_MAXLEN)
        dev_x[i] = sequence.pad_sequences(dev_x[i], MC.PAD_MAXLEN)
        test_x[i] = sequence.pad_sequences(test_x[i], MC.PAD_MAXLEN)
        w = MC.VOCAB_SIZE
        if i=='tags':
            w = MC.TAG_SIZE
        elif i=='code':
            w = MC.CODE_SIZE
        train_x[i] = train_x[i]*(train_x[i]<w)
        dev_x[i] = dev_x[i]*(dev_x[i]<w)
        test_x[i] = test_x[i]*(test_x[i]<w)

    train_y = [score_to_rank(i) for i in train_y]
    dev_y = [score_to_rank(i) for i in dev_y]
    test_y = [score_to_rank(i) for i in test_y]

    overal_maxlen = {}
    for i in ['content','code','tags','title']:
        overal_maxlen[i] = max(train_maxlen[i], dev_maxlen[i], test_maxlen[i])

    return ((train_x, train_y), (dev_x, dev_y), (test_x, test_y), vocab, len(vocab['content']), overal_maxlen, 1)
