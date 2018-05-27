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

mnist_data = tf.contrib.learn.datasets.load_dataset("mnist")

LENGTH     = 28 # side length of image
batch_size = 20

def get_one_hot_vector(batch_ys):
	output = []
	for i in batch_ys:
		temp    = np.zeros(10)
		temp[i] = 1
		output.append(temp)
	return np.asarray(output, dtype=np.float32)

img_train  = tf.placeholder(tf.float32, shape=[None, LENGTH, LENGTH, 1], name='img_input')
img_label  = tf.placeholder(tf.float32, shape=[None, 10], name="img_label")
conv1      = tf.layers.conv2d(img_train, 64, 5, name='img_conv1', activation=tf.nn.relu)
conv1_     = tf.layers.flatten(conv1)
fc         = tf.layers.dense(conv1_, 10, activation=tf.nn.softmax)

loss       = tf.reduce_mean(-img_label*tf.log(fc))
loss_graph = tf.summary.scalar("loss", loss)
optim      = tf.train.AdamOptimizer().minimize(loss)

epochs     = 5000

sess       = tf.Session()
sess.run(tf.global_variables_initializer())

saver      = tf.train.Saver()
graph_data = tf.summary.merge_all()
writer     = tf.summary.FileWriter('./logs', sess.graph)

# don't save any models. each run should start afresh

print("Training on images")

for i in range(epochs):
	batch_xs, batch_ys = mnist_data.train.next_batch(batch_size)
	batch_xs           = np.reshape(batch_xs, (batch_size, LENGTH, LENGTH, 1), order='C')
	batch_ys           = get_one_hot_vector(batch_ys)
	_, l, l_graph      = sess.run([optim, loss, loss_graph], feed_dict= {img_train: batch_xs, img_label: batch_ys})
	writer.add_summary(l_graph, i)
	print "Epoch {}/{} Loss {}".format(i, epochs, l)
	if (i % 100 == 1):
		saver.save(sess, save_path='./models/model_', global_step=i)

sess.close()