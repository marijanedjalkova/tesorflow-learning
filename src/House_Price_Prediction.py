import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation


num_house = 160
np.random.seed(42)
# create an array of size 160 with random values between 1000 and 3500
house_size = np.random.randint(low=1000, high=3500, size=num_house)

np.random.seed(42)
# create an array of size 160 with 100*house size + random values between 2000 and 70000
# we set the correlation here: the price tends to be 100 times larger than the house size
house_price = house_size * 100.0 + np.random.randint(low=2000, high=70000, size=num_house)


def show_initial_data():
    plt.plot(house_size, house_price, "bx")  # bx = blue
    plt.ylabel("Price")
    plt.show()


show_initial_data()
print("Mean house price: ", house_price.mean())
print("Standard house price: ", house_price.std())


def normalise(array):
    return (array-array.mean()) / array.std()


# define number of samples - 70% of all data.
# the other 30% will be the testing data
num_train_samples = math.floor(num_house * 0.7)

# define training data - cut off 70%
train_house_size = np.asarray(house_size[:num_train_samples])
train_price = np.asarray(house_price[:num_train_samples])

train_house_size_norm = normalise(train_house_size)
train_price_norm = normalise(train_price)

# define test data - the rest 30%
test_house_size = np.array(house_size[num_train_samples:])
test_house_price = np.array(house_price[num_train_samples:])

test_house_size_norm = normalise(test_house_size)
test_house_price_norm = normalise(test_house_price)

# ALL THE TRAINING AND TESTING DATA IS READY
# FOLLOWING IS THE TENSORFLOW SETUP

tf_house_size = tf.placeholder("float", name="house_size")
tf_price = tf.placeholder("float", name="price")

# the values of these will change as training progresses
# these are the things we are trying to predict, really
tf_size_factor = tf.Variable(np.random.random(), name="size_factor")
tf_price_offset = tf.Variable(np.random.random(), name="price_offset")


# this function shows that we know we are trying to predict size factor and price offset
# we know that the formula looks roughly like this:
# ( house_size * size_factor ) + price_offset
tf_price_pred = tf.add(tf.multiply(tf_size_factor, tf_house_size), tf_price_offset)


# loss function: how much error - mean squared error
# (tf_price_pred - tf_price) ^ 2 / ( 2 * num_train_samples)
# 2 in the denominator is a tensor itself
tf_cost = tf.reduce_sum(tf.pow(tf_price_pred - tf_price, 2))/(2*num_train_samples)

# optimiser learning rate
learning_rate = 0.1

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_cost)

init = tf.global_variables_initializer()


def plot_result(session):
    train_house_size_mean = train_house_size.mean()
    train_house_size_std = train_house_size.std()

    train_price_mean = train_price.mean()
    train_price_std = train_price.std()

    plt.rcParams["figure.figsize"] = (10,8)
    plt.figure()
    plt.ylabel("Price")
    plt.xlabel("Size (sq.ft)")
    plt.plot(train_house_size, train_price, 'go', label="Training data")
    plt.plot(test_house_size, test_house_price, 'mo', label="Testing data")
    plt.plot(train_house_size_norm * train_house_size_std + train_house_size_mean,
             (session.run(tf_size_factor) * train_house_size_norm + session.run(tf_price_offset))
             * train_price_std + train_price_mean, label="Learned regression")

    plt.legend(loc="upper left")
    plt.show()


def train():
    # launch the graph in the session
    with tf.Session() as sess:
        sess.run(init)

        display_every = 2
        num_training_iter = 50


        # do the following 50 times
        for iteration in range(num_training_iter):

            # iterate through training data
            for (x, y) in zip(train_house_size_norm, train_price_norm):
                # feed the training data to the optimiser
                sess.run(optimizer, feed_dict={tf_house_size: x, tf_price: y})

            if (iteration + 1) % display_every == 0:
                c = sess.run(tf_cost, feed_dict={tf_house_size: train_house_size_norm, tf_price: train_price_norm})
                print("iteration #: ", '%04d' % (iteration + 1), "cost=", "{:.9f}".format(c),
                      "size_factor=", sess.run(tf_size_factor), "price_offset=", sess.run(tf_price_offset))

        print("Optimisation finished")
        training_cost = sess.run(tf_cost, feed_dict={tf_house_size: train_house_size_norm, tf_price: train_price_norm})
        print("Training cost=", training_cost, "size_factor=", sess.run(tf_size_factor), "price_offset=",
              sess.run(tf_price_offset))
        plot_result(sess)


train()

