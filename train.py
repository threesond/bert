from config import Config as cfg
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from attn_model import net_atten as net
import sounddevice as sd
from audio_util import revert_vocoder
from utils import save_graph

def get_data():
    index = np.random.randint(0,len(train_data))
    data = train_data[index]
    features = data[:,:-1]
    phones = data[:,-1]
    return phones, features.T

tf.reset_default_graph()
sess = tf.Session()

train_data_path = cfg.train_dir+'/train_vocoder.npy'
train_data = np.load(train_data_path).item()['data']
train_data = train_data[:1]

input_phone = tf.placeholder(tf.int32,[None],name='input')
input_features = tf.placeholder(tf.float32,[cfg.sp_dim+2,None])
input_pos = tf.placeholder(tf.int32,[None],name='pos')
logit = net(input_phone,input_pos)

loss_op = tf.reduce_mean(tf.abs(logit - input_features))
train_op = tf.train.AdamOptimizer(0.0001).minimize(loss_op)

sess.run(tf.global_variables_initializer())

while True:
    temp_phone, temp_features = get_data()
    temp_pos = np.arange(0,len(temp_phone))
    feed = {input_phone:temp_phone,input_features:temp_features,input_pos:temp_pos}
    _,loss_value = sess.run([train_op,loss_op],feed)
    print (loss_value)
