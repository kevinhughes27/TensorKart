#!/usr/bin/env python

from utils import Data
import model
import tensorflow as tf
import numpy as np

# Load Training Data
data = Data("data/train")
val  = Data("data/validate").ret_n(500)

# Start session
sess = tf.InteractiveSession()

# Learning Functions
L2NormConst = 0.001
train_vars = tf.trainable_variables()
loss = tf.reduce_mean(tf.square(tf.sub(model.y_, model.y))) + tf.add_n([tf.nn.l2_loss(v) for v in train_vars]) * L2NormConst
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

sess.run(tf.global_variables_initializer())


# Training loop variables
epochs = 100
batch_size = 50
num_samples = data.num_examples
step_size = int(num_samples / batch_size)


early_stop_steps = 200
best_score       = np.inf
val_loss_value   = np.inf
best_idx         = 0
stop_flag        = False

for epoch in range(epochs):
    if stop_flag:
        break

    for i in range(step_size):
        batch = data.next_batch(batch_size)
        cur_step = epoch * step_size + i


        train_step.run(feed_dict={model.x: batch[0], model.y_: batch[1], model.keep_prob: 0.8})

        ## Update early stopping params
        if val_loss_value <= best_score:
          best_idx   = epoch * step_size + i
          best_score = val_loss_value


        if i%10 == 0:
          train_loss_value = loss.eval(feed_dict={model.x:batch[0], model.y_: batch[1], model.keep_prob: 1.0})
          val_loss_value   = loss.eval(feed_dict={model.x:val[0], model.y_:val[1], model.keep_prob: 1.0})
          print("epoch: %d step: %d train_loss: %g val_loss: %g best_idx: %g best_score: %g"%
            (epoch, cur_step, train_loss_value, val_loss_value, best_idx, best_score))

        ## Exit loop if early_stopping criteria is met
        if (cur_step - best_idx) >= early_stop_steps:
          stop_flag = True
          break



# Save the Model
saver = tf.train.Saver()
saver.save(sess, "model.ckpt")

