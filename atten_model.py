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

def embed(inputs,ori_dim,emb_dim, reuse=None):
    with tf.variable_scope('embed_'+get_uid(), reuse=reuse):
        lookup_table = tf.get_variable('lookup_table', 
                                       dtype=tf.float32, 
                                       shape=[ori_dim, emb_dim],
                                       initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
        outputs = tf.nn.embedding_lookup(lookup_table, inputs)
        return outputs

def dummy_func(input_data):
    return input_data   

def get_uid():
    return str(uuid.uuid4())[:8]

def linear(input_data,output_dim,act):
    with tf.variable_scope('linear_'+get_uid()):
        var = make_variable([input_data.get_shape()[1].value,output_dim],'var')
        output = tf.einsum('ij,jk->ik',input_data,var)
        bias = make_variable([1,output_dim],'bias')
        output = output + bias
        output = act(output)
        return output

def transpose_for_scores(input_tensor,head_num,key_size):
    output_tensor = tf.reshape(input_tensor, [-1, head_num, key_size])
    output_tensor = tf.transpose(output_tensor, [1,0,2])
    return output_tensor

def mha_block2(input_data,key_size,head_num):
    with tf.variable_scope('mha_'+get_uid()):
        query_layer = linear(input_data,key_size*head_num,dummy_func)
        key_layer = linear(input_data,key_size*head_num,dummy_func)
        value_layer = linear(input_data,key_size*head_num,dummy_func)
        query_layer = transpose_for_scores(query_layer,head_num,key_size)
        key_layer = transpose_for_scores(key_layer,head_num,key_size)
        query_layer = tf.expand_dims(query_layer,0)
        key_layer = tf.expand_dims(key_layer,0)
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        attention_scores = tf.multiply(attention_scores,
                                 1.0 / tf.sqrt(float(key_size)))
        attention_probs = tf.nn.softmax(attention_scores)
        value_layer = tf.reshape(value_layer,[1,-1,head_num,key_size])
        value_layer = tf.transpose(value_layer, [0, 2, 1, 3])
        context_layer = tf.matmul(attention_probs, value_layer)
        context_layer = tf.transpose(context_layer, [0, 2, 1, 3])
        context_layer = tf.reshape(context_layer,[-1,key_size*head_num])
        return context_layer
          

def net_atten(input_data,input_pos,reuse=False):
    with tf.variable_scope('net',reuse=reuse):
        x = embed(input_data,cfg.vocab_size,cfg.vocab_dim)
        pos = embed(input_pos,800,cfg.vocab_dim)
        x = x + pos
        x = normalize(x,'norm_'+get_uid())
        for _ in range(3):
            short_cut = x
            x = mha_block2(x,64,8)
            x = linear(x,512,dummy_func)
            x = normalize(x+short_cut,'norm_'+get_uid())
            ix = linear(x,512,tf.nn.relu)
            lx = linear(ix,512,dummy_func)
            x = normalize(lx+x,'norm_'+get_uid())
        x = linear(x,27,dummy_func)
        output = tf.transpose(x)
        return output
