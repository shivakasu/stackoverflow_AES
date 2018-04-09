#!/usr/bin/env python

import numpy as np
from keras.models import *
from .AES.nea.reader import *
from keras.layers import *
from .AES.nea.layers import *
from .AES.nea.config import ModelConfig as MC
from .AES.nea.self_attention import *
from keras.preprocessing import sequence
from sklearn import preprocessing
import nltk

class PretrainedModel:

    json_file = open('aes/AES/model.json','r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json,custom_objects={'Attention':Attention})
    model.load_weights('aes/AES/output_dir/final_2gru_20_20_20_cross.h5')
    attmodel = Model(inputs=model.input, outputs=model.get_layer('concatenate_1').output)
    #  a bug in keras ===> https://zhuanlan.zhihu.com/p/27101000
    bug = [np.array([np.zeros(300)]),np.array([np.zeros(300)]),np.array([np.zeros(300)]),np.array([np.zeros(300)])]
    model.predict(bug)
    attmodel.predict(bug)

    @classmethod
    def predict(cls,data):

        #remember '\n' in origin text
        con_br = []
        cod_br = []
        content1 = str_expand(data['content'].lower())
        code1 = str_expand(data['code'].lower())
        content2 = re.sub(r"[\t\r\f ]+", " ", content1)
        code2 = re.sub(r"[\t\r\f ]+", " ", code1)
        content1 = nltk.word_tokenize(re.sub(r"[\n]+", "qwerqwer", content2))
        code1 = code_split(code2.replace('\n','qwerqwer'))
        for i in range(len(content1)):
            if 'qwerqwer' == content1[i]:
                con_br.append(i-len(con_br)-1)
        for i in range(len(code1)):
            if 'qwerqwer' == code1[i]:
                cod_br.append(i-len(cod_br)-1)
        content1 = [i for i in content1 if i!='qwerqwer']
        code1 = [i for i in code1 if i!='qwerqwer']
        for i in range(len(content1)):
            if 'qwerqwer' in content1[i]:
                con_br.append(i)
        for i in range(len(code1)):
            if 'qwerqwer' in code1[i]:
                cod_br.append(i)
                
        content1 = tokenize(content2.replace('\n',' '))
        title1 = tokenize(data['title'].lower())
        code1 = code_split(code2.replace('\n',' '))
        vocab = load_vocab(MC.VOCAB_PATH)
        content = vectorize(vocab['content'],content1)
        code = vectorize(vocab['code'],code1)
        title = vectorize(vocab['content'],title1)
        tags = vectorize(vocab['tags'],data['tag'].split(' '))
        content = [i if i<MC.VOCAB_SIZE else 0 for i in content]
        code = [i if i<MC.CODE_SIZE else 0 for i in code]
        title = [i if i<MC.VOCAB_SIZE else 0 for i in title]
        tags = [i if i<MC.TAG_SIZE else 0 for i in tags]
        content = sequence.pad_sequences([content],MC.PAD_MAXLEN)
        code = sequence.pad_sequences([code],MC.PAD_MAXLEN)
        title = sequence.pad_sequences([title],MC.PAD_MAXLEN)
        tags = sequence.pad_sequences([tags],MC.PAD_MAXLEN)
        score = cls.model.predict([content,code,title,tags]).squeeze()
        att = cls.attmodel.predict([content,code,title,tags]).squeeze()
        con_att = preprocessing.minmax_scale(att[0:len(content1)])
        cod_att = preprocessing.minmax_scale(att[600:600+len(code1)])
        tit_att = preprocessing.minmax_scale(att[1200:1200+len(title1)])
        res = {'conbr':con_br,'codbr':cod_br,'score':score.tolist(),'content':content1,'code':code1,'title':title1,'con_att':con_att.tolist(),'cod_att':cod_att.tolist(),'tit_att':tit_att.tolist()}
        return res
