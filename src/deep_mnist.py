import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# with InteractiveSession do not need to pass sess
sess = tf.InteractiveSession()

# placeholder for the 28x28 image data
x = tf.placeholder(tf.float32, shape=[None, 784])

# vector for predicted probabilities for each
# digit(0-9) class.
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# shape is now 28x28 and 0-1 for greyscale
x_image = tf.reshape(x, [-1,28,28,1], name="x_image")

# create not only weights and biases but also convolution,
# pooling layers.
# RELU - 0 if <0, otherwise y
def weightvariable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# convolution
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

# pooling
# ksize - kernel size
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1],
                          padding='SAME')

# define layers

# 1st layer
# 32 features for each 5x5 patch of the image
W_conv1 = weightvariable([5,5,1,32])
b_conv1 = bias_variable([32])

# do convolution on images, add bias, push through relu activation
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# take results and run through the max pool
h_pool1 = max_pool_2x2(h_conv1)

# 2nd layer

# Process the 32 features from layer1, in 5x5 patch.
# return 64 features weights and biases.
W_conv2 = weightvariable([5,5,32,64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Fully connected layer
W_fc1 = weightvariable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

# connect output of pooling layer 2 as input to fully connected layer
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout some neurons to reduce overfitting
keep_prob = tf.placeholder(tf.float32) # get dropout probability as a training input
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob) # tuning parameter

# readout layer
W_fc2 = weightvariable([1024, 10])
b_fc2 = bias_variable([10])

# define model
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=y_conv, labels=y_))

# loss optimisation
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.global_variables_initializer())

import time
num_steps = 3000
display_every=100

start_time = time.time()
end_time = time.time()

for i in range(num_steps):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x: batch[0], y_:batch[1], keep_prob:0.5})

    if i%display_every == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], y_:batch[1], keep_prob:1.0})
        end_time = time.time()
        print("step {0}, time {1:.2f}s, training accuracy {2:.3f}".format(
            i, end_time-start_time, train_accuracy*100.0))

end_time = time.time()
print("Total training time for {0} batches: {1:.2f}s".format(
    i+1, end_time-start_time))

print("Test accuracy {0:.3f}".format(
    accuracy.eval(feed_dict={x:mnist.test.images,
                             y_: mnist.test.labels, keep_prob:1.0})*100.0))
sess.close()










