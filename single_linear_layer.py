#!/usr/bin/env python2.7

import libs.orlfaces as lo
import tensorflow as tf
import pprint as pp
import sys


# "The role of the Python code is therefore to build this external computation graph,
# and to dictate which parts of the computation graph should be run"


# orlfaces is a class which store training, validation and testing sets as numpy arrays.
orlfaces = lo.orlfaces_loader(sys.argv[1:])
print orlfaces.train.num_inputs, orlfaces.train.num_classes

# A Session is use to  execute ops in the graph.
# Here we use an InteractiveSession to gain flexibility
# aka 'interleave operations which build a computation graph with ones that run the graph'
sess = tf.InteractiveSession()

# We create x and y_ placeholder, which will be later fed when we'll ask
# TensorFlow to run a computation.
# The input (x) will be a 2d tensor of floating point number. The shape parameter
# is not mandatory but we set it to catch bugs related to inconsistent tensor shapes.
# "orlfaces.train.num_inputs is the dimensionality of a single flattened ORLFACES image, and None indicate that
# the first dimension, aka the batch size, can be of any size"
x = tf.placeholder("float", shape=[None, orlfaces.train.num_inputs])
# "The output classes (y_) will also consist of a 2d tensor, where each row is a one-hot 2-dimensional vector"
y_ = tf.placeholder("float", shape=[None, orlfaces.train.num_classes])

# We define the weights W and biases b for our model. To handle it we use Variable,
# which "is a value that lives in TF computation graph. It can be used and even
# modified by the computation"
# Both parameter are initialize as tensor full of zeros.
# "W is a orlfaces.train.num_inputs x orlfaces.train.num_classes matrix (because we have orlfaces.train.num_inputs input features and orlfaces.train.num_classes outputs), and b is
# a orlfaces.train.num_classes-dimentional vector (because we have orlfaces.train.num_classes classes).
W = tf.Variable(tf.zeros([orlfaces.train.num_inputs, orlfaces.train.num_classes]))
b = tf.Variable(tf.zeros([orlfaces.train.num_classes]))

# To use Variable within a session, they must be initialized using that session.
# "This step takes the initial values (in this case tensors full of zeros) that have
# already been specified, and assigns them to each Variable. This can be done for all
# Variables at once."
sess.run(tf.initialize_all_variables())

# Implementation of the regression model. We multiply the images (where each images is a vector
# since we flatten it).
# So we multiply "the input images x by the weight matrix W, add the bias b and compute the softmax
# probabilities that are assigned to each class."
y = tf.nn.softmax(tf.matmul(x, W) + b)

# The cost function to be minimized during training can be specified just as easily.
# Our cost function will be the cross-entropy between the target and the model's prediction.
cross_entropy = -tf.reduce_sum(y_ * tf.log(y)) # should be change --> http://stackoverflow.com/questions/33712178/tensorflow-nan-bug

# Now that the model is describe, we can use TensorFlow to train it.
# "For this example, we will use steepest gradient descent, with a step length of 0.01,
# to descend the cross entropy."
# So this line will "add new operations to the computation graph.
# These operations included ones to compute gradients, compute parameter update steps,
# and apply update steps to the parameters."
train_step = tf.train.GradientDescentOptimizer(1e-30).minimize(cross_entropy)

# "The returned operation train_step, when run, will apply the gradient descent updates to the parameters.
# Training the model can therefore be accomplished by repeatedly running train_step."
for i in xrange(10000):
    batch = orlfaces.train.next_batch(3)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})
    if i % 100 == 0:
        print cross_entropy.eval({x: orlfaces.train.images, y_: orlfaces.train.labels}, sess)


# We can now check the result of our model.
# tf.armax "gives you the index of the highest entry in a tensor along some axis and
# tf.equal "check if our prediction matches the truth."
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print accuracy.eval(feed_dict = {x: orlfaces.test.images, y_ : orlfaces.test.labels})
