# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 15:42:33 2018

@author: 天降大酱
"""

import tensorflow as tf
from ops import conv1d, make_variable, normalize, conv2d, conv1ds, conv1d_transpose, conv1dc
from config import Config as cfg
import numpy as np
import uuid

#嵌入
def embed(inputs,ori_dim,emb_dim, reuse=None):
    with tf.variable_scope('embed_'+get_uid(), reuse=reuse):
        lookup_table = tf.get_variable('lookup_table', 
                                       dtype=tf.float32, 
                                       shape=[ori_dim, emb_dim],
                                       initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
        outputs = tf.nn.embedding_lookup(lookup_table, inputs)
        return outputs

#什么都不做的function
def dummy_func(input_data):
    return input_data   

#得到一个uid
def get_uid():
    return str(uuid.uuid4())[:8]

#矩阵乘之后加上个bias,然后再作用一个激活函数act
def linear(input_data,output_dim,act):
    with tf.variable_scope('linear_'+get_uid()):
        var = make_variable([input_data.get_shape()[1].value,output_dim],'var')
        output = tf.einsum('ij,jk->ik',input_data,var)
        bias = make_variable([1,output_dim],'bias')
        output = output + bias
        output = act(output)
        return output

#reshape和transpose(从google的bert里面抄的)
def transpose_for_scores(input_tensor,head_num,key_size):
    output_tensor = tf.reshape(input_tensor, [-1, head_num, key_size])
    output_tensor = tf.transpose(output_tensor, [1,0,2])
    return output_tensor

#multi headed attention(大部分从google抄的)
#请参考 https://github.com/google-research/bert
def mha_block2(input_data,key_size,head_num):
    with tf.variable_scope('mha_'+get_uid()):
        #先从input_data得到query,key,value
        query_layer = linear(input_data,key_size*head_num,dummy_func)
        key_layer = linear(input_data,key_size*head_num,dummy_func)
        value_layer = linear(input_data,key_size*head_num,dummy_func)
        #再将上面的query,key拆分成head_num个以实现multi head attention
        query_layer = transpose_for_scores(query_layer,head_num,key_size)
        key_layer = transpose_for_scores(key_layer,head_num,key_size)
        #将query和key相乘得到需要作用于value的score
        query_layer = tf.expand_dims(query_layer,0)
        key_layer = tf.expand_dims(key_layer,0)
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        attention_scores = tf.multiply(attention_scores,
                                 1.0 / tf.sqrt(float(key_size)))
        attention_probs = tf.nn.softmax(attention_scores)
        #将score作用到value上
        value_layer = tf.reshape(value_layer,[1,-1,head_num,key_size])
        value_layer = tf.transpose(value_layer, [0, 2, 1, 3])
        context_layer = tf.matmul(attention_probs, value_layer)
        context_layer = tf.transpose(context_layer, [0, 2, 1, 3])
        context_layer = tf.reshape(context_layer,[-1,key_size*head_num])
        return context_layer
          

def net_atten(input_data,input_pos,reuse=False):
    with tf.variable_scope('net',reuse=reuse):
        #得到input_data的嵌入
        x = embed(input_data,cfg.vocab_size,cfg.vocab_dim)
        #得到输入位置input_pos的嵌入
        pos = embed(input_pos,800,cfg.vocab_dim)
        #将以上嵌入加在一起然后再做layernorm
        x = x + pos
        x = normalize(x,'norm_'+get_uid())
        #以下循环只要超过2次就会收敛的很慢
        for _ in range(3):
            #先预留一个short_cut做residue连接用
            short_cut = x
            #一个multi head attention 输出
            x = mha_block2(x,64,8)
            #矩阵乘到512,没有用激活函数
            x = linear(x,512,dummy_func)
            #给x+short_cut这个residue连接做layernorm
            x = normalize(x+short_cut,'norm_'+get_uid())
            #再加两个矩阵乘, 注意第一个有relu激活
            ix = linear(x,512,tf.nn.relu)
            lx = linear(ix,512,dummy_func)
            #最后给lx+x这个residue连接做layernorm
            x = normalize(lx+x,'norm_'+get_uid())
        #由于我拿声音特征做测试,所以输出为27维, 所以再做一个输出的矩阵乘
        x = linear(x,27,dummy_func)
        output = tf.transpose(x)
        return output
