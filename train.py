#!/usr/bin/env python

from utils import Data
import model
import tensorflow as tf

# Load Training Data
data = Data()

# Start session
sess = tf.InteractiveSession()

# Learning Functions
loss = tf.reduce_mean(tf.square(tf.sub(model.y, model.y_)))
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
sess.run(tf.global_variables_initializer())

# Train the Model
for i in range(20000):
  batch = data.next_batch(50)

  if i%100 == 0:
    train_accuracy = loss.eval(feed_dict={model.x:batch[0], model.y_: batch[1], model.keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))

  train_step.run(feed_dict={model.x: batch[0], model.y_: batch[1], model.keep_prob: 0.5})

# Save the Model
saver = tf.train.Saver()
saver.save(sess, "model.ckpt")
