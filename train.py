#!/usr/bin/env python

from utils import Data
import model
import tensorflow as tf

# Load Training Data
data = Data()

# Start session
sess = tf.InteractiveSession()

# Learning Functions
L2NormConst = 0.001
train_vars = tf.trainable_variables()
loss = tf.reduce_mean(tf.square(tf.subtract(model.y_, model.y))) + tf.add_n([tf.nn.l2_loss(v) for v in train_vars]) * L2NormConst
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

sess.run(tf.global_variables_initializer())

# Training loop variables
epochs = 100
batch_size = 50
num_samples = data.num_examples
step_size = int(num_samples / batch_size)

for epoch in range(epochs):
    for i in range(step_size):
        batch = data.next_batch(batch_size)

        train_step.run(feed_dict={model.x: batch[0], model.y_: batch[1], model.keep_prob: 0.8})

        if i%10 == 0:
          loss_value = loss.eval(feed_dict={model.x:batch[0], model.y_: batch[1], model.keep_prob: 1.0})
          print("epoch: %d step: %d loss: %g"%(epoch, epoch * batch_size + i, loss_value))

# Save the Model
saver = tf.train.Saver()
saver.save(sess, "model.ckpt")
