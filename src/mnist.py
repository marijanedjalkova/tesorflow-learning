import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# placeholder for the 28x28 image data
x = tf.placeholder(tf.float32, shape=[None, 784])

# vector for predicted probabilities for each
# digit(0-9) class.
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# weights and balances
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# placeholder is what we use for training data
# variable is what we use for predicting

# the model
y = tf.nn.softmax(tf.matmul(x, W) + b)

# loss is cross entropy
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# each training step in gradient descent we want to minimise cross entropy
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    # get 100 random data points from the data.
    # batch_xs = image, batch_ys = digit(0-9) class
    sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys})
    # do the optimisation with this data

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
test_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images,
                                              y_:mnist.test.labels})
print("Test accuracy: {}".format(test_accuracy* 100.0))
sess.close()





