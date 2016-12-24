#!/usr/bin/env python

import tensorflow as tf

IMG_W = 320
IMG_H = 240

IN_SHAPE = IMG_W*IMG_H
OUT_SHAPE = 5


# Start session
sess = tf.InteractiveSession()


# Placeholders
x = tf.placeholder(tf.float32, shape=[None, IN_SHAPE])
y_ = tf.placeholder(tf.float32, shape=[None, OUT_SHAPE])


# Weight Initialization
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


# Convolution and Pooling
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_6x6(x):
  return tf.nn.max_pool(x, ksize=[1, 6, 6, 1],
                        strides=[1, 6, 6, 1], padding='SAME')


# First Convolutional Layer
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,IMG_W,IMG_H,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_6x6(h_conv1)


# Second Convolutional Layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_6x6(h_conv2)

#print(h_pool2.get_shape()) # this is transfered into the next step


# Densely Connected Layer
W_fc1 = weight_variable([9 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 9*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


# Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


# Readout Layer
W_fc2 = weight_variable([1024, OUT_SHAPE])
b_fc2 = bias_variable([OUT_SHAPE])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


saver = tf.train.Saver()
saver.restore(sess, "./model.ckpt")


import numpy as np
from skimage.color import rgb2gray
import wx
wx.App()


# Play
def take_screenshot():
    screen = wx.ScreenDC()
    size = screen.GetSize()
    bmp = wx.Bitmap(size[0], size[1])
    mem = wx.MemoryDC(bmp)
    mem.Blit(0, 0, size[0], size[1], screen, 0, 0)
    return bmp.GetSubBitmap(wx.Rect([0,0],[640,480]))


# init joystick
import uinput
device = uinput.Device([
    uinput.ABS_X,
    uinput.ABS_Y,
    uinput.BTN_SOUTH, # a
    uinput.BTN_NORTH, # x
    uinput.BTN_TR # rb
])


# button mappings determined using evtest
def send_joystick(output):
    import pdb; pdb.set_trace()

    # I have to calibrate the driver
    # my training data was between -1 and 1
    # this is between different ranges

    device.emit(uinput.ABS_X, output[0])
    device.emit(uinput.ABS_Y, output[1])
    device.emit(uinput.BTN_SOUTH, output[2])
    device.emit(uinput.BTN_NORTH, output[3])
    device.emit(uinput.BTN_TR, output[4])


from skimage import transform

while True:
    ## Look
    bmp = take_screenshot()
    buf = bmp.ConvertToImage().GetData()
    image = np.frombuffer(buf, dtype='uint8')
    image = image.reshape(480, 640, 3)
    image = rgb2gray(image)
    image = transform.resize(image, [IMG_H, IMG_W])
    in_vec = image.flatten()

    ## Think
    joystick_output = y_conv.eval(feed_dict={x: [in_vec], keep_prob: 1.0})[0]

    ## Act
    # need to make sure mupen64plus is listening to my fake controller input
    send_joystick(joystick_output)

    ## Shadow (display action but don't act)
