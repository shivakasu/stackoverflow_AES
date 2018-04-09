from collections.abc import Callable, Sequence
import numpy as np
from keras import backend as K
from keras import initializers
from keras import activations
from keras.engine.topology import Layer
from .mixins import MergfuncMixin

from keras.layers import merge
from keras.layers.core import *
from keras.models import *

def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(TIME_STEPS, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    return output_attention_mul


class Attention(Layer):
    def __init__(self, op='attsum', activation='tanh', init_stdev=0.01, **kwargs):
        self.supports_masking = True
        assert op in {'attsum', 'attmean'}
        assert activation in {None, 'tanh'}
        self.op = op
        self.activation = activation
        self.init_stdev = init_stdev
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        init_val_v = (np.random.randn(input_shape[2]) * self.init_stdev).astype(K.floatx())
        self.att_v = K.variable(init_val_v, name='att_v')
        init_val_W = (np.random.randn(input_shape[2], input_shape[2]) * self.init_stdev).astype(K.floatx())
        self.att_W = K.variable(init_val_W, name='att_W')
        self.trainable_weights = [self.att_v, self.att_W]

    def call(self, x, mask=None):
        y = K.dot(x, self.att_W)
        if not self.activation:
            weights = K.tf.tensordot(self.att_v, y, axes=[[0], [2]])
        elif self.activation == 'tanh':
            weights = K.tf.tensordot(self.att_v, K.tanh(y), axes=[[0], [2]])
        weights = K.softmax(weights)
        out = x * K.permute_dimensions(K.repeat(weights, x.shape[2]), [0, 2, 1])
        if self.op == 'attsum':
            out = K.sum(out,axis=1)
        elif self.op == 'attmean':
            out = K.sum(out,axis=1) / K.sum(out,axis=1, keepdims=True)
        return K.cast(out, K.floatx())

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])

    def compute_mask(self, x, mask):
        return None

    def get_config(self):
        config = {'op': self.op, 'activation': self.activation, 'init_stdev': self.init_stdev}
        base_config = super(Attention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Attention1D(Layer, MergfuncMixin):
    """self-attention的特点是自己为输入,输出也是一个和自己一样shape的张量.

    Attributes:
        similarity (Union[Callable,str]): - 指定使用的相似度计算函数,目前可选的有\
        加性相似度(additive),乘性相似度(multiplicative),点乘相似度(dot_product),\
        当然也可以自己写一个,只要最终输出的是一个(?,output,input_timestep)的

        mergfunc (Union[Callable,str]): - 与similarity类似,目前可选的有:\
        矩阵乘法(batch_dot_merg),逐项相乘(batch_mul_merg),逐项相加(batch_add_merg)

        kernel_size (tuple[int,int]): - 指定使用加性相似度(additive)时才能指定,\
        用于指定第一个权重的形状,各维的意义[输出的纬度,第二个权重矩阵的第一个纬度]

        kernel_initializer (str): - 第一个权重的初始化函数,默认glorot_uniform

        wk_kernel_initializer (str): - 第二个权重的初始化函数,默认glorot_uniform
    """

    def __init__(self, similarity="linear", *,
                 mergfunc=None,
                 kernel_size=None,
                 dropout_rate=None,
                 kernel_initializer='glorot_uniform',
                 wk_kernel_initializer='glorot_uniform',
                 **kwargs):
        if isinstance(similarity, Callable):
            self.similarity = similarity
        elif isinstance(similarity, str) and similarity in (
                "multiplicative", "dot_product", "additive", "linear"):
            self.similarity = similarity
        else:
            raise ValueError(
                'similarity now only support '
                '"multiplicative","dot_product","additive",'
                'and you can input a function as the similarity function!'
            )
        if similarity == "additive" and kernel_size is None:
            raise ValueError(
                'additive similarity need '
                'hyperparameter kernel_size!'
            )
        if similarity != "additive" and kernel_size:
            print(kernel_size)
            print(
                'only additive similarity support '
                'hyperparameter kernel_size!'
            )
            kernel_size = None

        if (isinstance(
            kernel_size,
                Sequence) and len(kernel_size) == 2) or kernel_size is None:
            self.kernel_size = kernel_size
        else:
            raise ValueError(
                'kernel_size must be a Sequence with 2 int element')
        self.dropout_rate = dropout_rate
        self.mergfunc = mergfunc
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.wk_kernel_initializer = initializers.get(
            wk_kernel_initializer)
        self.dim = None
        if self.similarity == "additive":
            self.kernel = None
            self.wk_kernel = None
            self.wq_kernel = None
        elif self.similarity == "multiplicative":
            self.kernel = None
        super().__init__(**kwargs)

    def _build_w(self, time, dim):
        self.dim = dim
        if self.similarity == "additive":
            r, d_a = self.kernel_size
            self.kernel = self.add_weight(name='kernel',
                                          shape=(r, d_a),
                                          initializer=self.kernel_initializer
                                          )

            self.wk_kernel = self.add_weight(
                name='wk_kernel',
                shape=(d_a, dim),
                initializer=self.wk_kernel_initializer)
        elif self.similarity == "multiplicative":
            self.kernel_size = (time, dim)
            self.kernel = self.add_weight(name='kernel',
                                          shape=(dim, dim),
                                          initializer=self.kernel_initializer)
        else:
            self.kernel_size = (time, dim)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError('A additive weight layer should be called '
                             'by a (batch,time_step,dim)3D inputs.'
                             'Got ' + str(input_shape) + ' inputs.')

        time = input_shape[-2]
        dim = input_shape[-1]
        self._build_w(time, dim)
        # Be sure to call this somewhere!
        super().build(input_shape)

    def multiplicative(self, Source):
        r"""乘性相似度,其中的权重矩阵形状为[dim,dim]\
        输出的固定为与原输入一样形状

        .. math::  Similarity(Source) =  Source\cdot W \cdot Source^T
        """
        Source_t = K.permute_dimensions(Source, (0, 2, 1))
        s = K.dot(Source, self.kernel)
        sim = K.batch_dot(s, Source_t)
        return sim

    def dot_product(self, Source):
        r"""点乘相似度,在google的attention is all you need 中看到的.\
        很迷,没有要训练的矩阵,直接转置点乘,输出的固定为与原输入一样形状

        .. math::  Similarity(Source) = \frac{Source^T\cdot Source}{\sqrt{d_k}}
        """
        Source_t = K.permute_dimensions(Source, (0, 2, 1))
        sim = K.batch_dot(Source, Source_t)
        return sim

    def additive(self, Source):
        r"""
        加性相似度,最经典的注意力相似度机制,如果是在self attention中\
        则该层有两个权重矩阵形状为(r,d_a)和(d_a,dim)

        .. math::  Similarity(Source) = V \cdot tanh(W_k\cdot Source^T)
        """
        Source_t = K.permute_dimensions(Source, (0, 2, 1))
        f_att = K.dot(self.wk_kernel, Source_t)
        f_att = K.permute_dimensions(f_att, (1, 0, 2))
        sim = K.dot(self.kernel, K.tanh(f_att))
        sim = K.permute_dimensions(sim, (1, 0, 2))
        return sim

    def linear(self, Source):
        self.mergfunc = "batch_mul_merg"
        Source_t = K.permute_dimensions(Source, (0, 2, 1))
        return Source_t

    def _call_attention(self, Source):
        r"""self-attention就是通过相似度函数计算得的相似矩阵过softmax后与自身点乘得到

        .. math::  A = Softmax(Similarity(Source))
        .. math::  C = mergfunc(A,Source)
        """
        if isinstance(self.similarity, Callable):
            sim = self.similarity(Source)
        else:
            sim = getattr(self, self.similarity)(Source)

        sm = activations.softmax(sim)
        if self.dropout_rate:
            sm = K.dropout(sm, self.dropout_rate)
        if isinstance(self.mergfunc, Callable):
            result = self.mergfunc(sm, Source)
        elif isinstance(self.mergfunc, str):
            result = getattr(self, self.mergfunc, 'batch_dot_merg')(sm, Source)
        else:
            result = getattr(self, 'batch_dot_merg')(sm, Source)
        return result

    def call(self, inputs):
        Source = inputs
        result = self._call_attention(Source)
        return result

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])
        # return (input_shape[0], self.kernel_size[0], self.dim)

    def get_config(self):
        config = {
            'similarity': self.similarity,
            'mergfunc': self.mergfunc,
            'kernel_size': self.kernel_size,
            'dropout_rate': self.dropout_rate,
            'kernel_initializer': self.kernel_initializer,
            'wk_kernel_initializer': self.wk_kernel_initializer
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
