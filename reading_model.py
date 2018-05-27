# Reference

'''
tf.layers.conv2d
----------------

tf.layers.conv2d(
	inputs,
	filters,
	kernel_size,
	strides=(1, 1),
	padding='valid',
	data_format='channels_last',
	dilation_rate=(1, 1),
	activation=None,
	use_bias=True,
	kernel_initializer=None,
	bias_initializer=tf.zeros_initializer(),
	kernel_regularizer=None,
	bias_regularizer=None,
	activity_regularizer=None,
	kernel_constraint=None,
	bias_constraint=None,
	trainable=True,
	name=None,
	reuse=None
)

tf.layers.dense
---------------

tf.layers.dense(
	inputs,
	units,
	activation=None,
	use_bias=True,
	kernel_initializer=None,
	bias_initializer=tf.zeros_initializer(),
	kernel_regularizer=None,
	bias_regularizer=None,
	activity_regularizer=None,
	kernel_constraint=None,
	bias_constraint=None,
	trainable=True,
	name=None,
	reuse=None
)
'''
import numpy as np
import tensorflow as tf
import scipy.misc as sp
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import argparse, sys
import os

parser   = argparse.ArgumentParser()

parser.add_argument('--cmap', help='The kind of cmap to be used')
args     = parser.parse_args()

cmap_val = args.cmap

if not cmap_val:
	folder_name = 'cmap'
else:
	folder_name = 'grey'

mnist_data = tf.contrib.learn.datasets.load_dataset("mnist")

batch_size = 10
LENGTH     = 28

img_train  = tf.placeholder(tf.float32, shape=[None, LENGTH, LENGTH, 1], name='img_input')
img_label  = tf.placeholder(tf.float32, shape=[None, 10], name="img_label")
conv1      = tf.layers.conv2d(img_train, 64, 5, name='img_conv1', activation=tf.nn.relu)
conv1_     = tf.layers.flatten(conv1)
fc         = tf.layers.dense(conv1_, 10, activation=tf.nn.softmax)

load_checkpoint = 1
sess            = tf.Session()
sess.run(tf.global_variables_initializer())
saver           = tf.train.Saver()

if load_checkpoint:
	ckpt          = tf.train.get_checkpoint_state('./models')
	if ckpt and ckpt.model_checkpoint_path:
		ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
		saver.restore(sess, os.path.join('./models', ckpt_name))
		print "[*] Load success"
	else:
		print "[!] Load failure"


for i in range(10):
	batch_xs, _ = mnist_data.train.next_batch(batch_size)
	batch_xs    = np.reshape(batch_xs, (batch_size, LENGTH, LENGTH, 1), order='C')
	filters     = sess.run(conv1, feed_dict= {img_train: batch_xs})

	for j in range(batch_size):
		temp  = np.ones((8*25, 8*25))
		temp2 = np.zeros((24, 24))
		temp3 = batch_xs[j].reshape(28, 28)
		for k in range(64):
			f = filters[j, :, :, k].reshape((24, 24))
			temp[25*int(k/8):25*(int(k/8)+1)-1,
				25*(k%8):25*((k%8)+1)-1] = f
			temp2 += f
			print str(i)+'_'+str(j)+'_'+str(k)+'_'+str(_[j])
		plt.imshow(temp, cmap=cmap_val)
		plt.savefig('./filters_'+folder_name+'/'+str(_[j])+'_'+str(i)+str(j)+'.png')
		plt.clf()
		plt.imshow(temp2, cmap=cmap_val)
		plt.savefig('./filters_'+folder_name+'/'+str(_[j])+'_'+str(i)+str(j)+'_added.png')
		plt.clf()
		plt.imshow(temp3, cmap=cmap_val)
		plt.savefig('./filters_'+folder_name+'/'+str(_[j])+'_'+str(i)+str(j)+'_true.png')
		plt.clf()

sess.close()