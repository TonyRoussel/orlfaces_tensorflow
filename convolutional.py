#!/usr/bin/env python2.7

import libs.orlfaces as lo
import tensorflow as tf
import pprint as pp
import sys


# For this model we'll need "to create a lot of weights and biases"
# "One should generally initialize weights with a small amount of noise for 
# symmetry breaking, and to prevent 0 gradients. Since we're using ReLU neurons,
# it is also good practice to initialize them with a slightly positive
# initial bias to avoid "dead neurons.""
# "Instead of doing this repeatedly while we build the model, let's create
# two handy functions to do it for us"
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# "TensorFlow also gives us a lot of flexibility in convolution and pooling operations.
# How do we handle the boundaries? What is our stride size?
# In this example, we're always going to choose the vanilla version.
# Our convolutions uses a stride of one and are zero padded so that the output is the same size as the input.
# Our pooling is plain old max pooling over 2x2 blocks. To keep our code cleaner,
# let's also abstract those operations into functions."
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')



orlfaces = lo.orlfaces_loader(sys.argv[1:])
x = tf.placeholder("float", shape=[None, orlfaces.train.num_inputs])
y_ = tf.placeholder("float", shape=[None, orlfaces.train.num_classes])

print "AFTER PH DECLARATION" ########
print "x:", x.get_shape() #########
print "y_:", y_.get_shape() #########

# "We can now implement our first layer
# It will consist of convolution, followed by max pooling. The convolutional will compute
# 32 features for each 5x5 patch. Its weight tensor will have a shape of [5, 5, 1, 32].
# The first two dimensions are the patch size, the next is the number of input channels,
# and the last is the number of output channels. We will also have a bias vector with a
# component for each output channel."
# W_conv1 = weight_variable([patch_size_x, patch_size_y, num_input_channel, num_output_channel])
W_conv1 = weight_variable([5, 5, 1, 8])
b_conv1 = bias_variable([8])

# "To apply the layer, we first reshape x to a 4d tensor, with the second and third dimensions 
# corresponding to image width and height, and the final dimension corresponding to the number of color channels."
x_image = tf.reshape(x, [-1, 112, 92, 1])

# "We then convolve x_image with the weight tensor, add the bias, apply the ReLU function, and finally max pool."
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

print "AFTER FIRST LAYER DECLARATION" ########
print "W_conv1:", W_conv1.get_shape() #########
print "b_conv1:", b_conv1.get_shape() #########
print "x_img:", x_image.get_shape() #########
print "h_conv1:", h_conv1.get_shape() #########
print "h_pool1:", h_pool1.get_shape() #########

# "In order to build a deep network, we stack several layers of this type.
# The second layer will have 64 features for each 5x5 patch."
W_conv2 = weight_variable([5, 5, 8, 16])
b_conv2 = bias_variable([16])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

print "AFTER 2ND LAYER DECLARATION" ########
print "W_conv2:", W_conv2.get_shape() #########
print "b_conv2:", b_conv2.get_shape() #########
print "h_conv2:", h_conv2.get_shape() #########
print "h_pool2:", h_pool2.get_shape() #########


# "In order to build a deep network, we stack several layers of this type.
# The second layer will have 64 features for each 5x5 patch."
W_conv3 = weight_variable([5, 5, 16, 32])
b_conv3 = bias_variable([32])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

print "AFTER 3RD LAYER DECLARATION" ########
print "W_conv3:", W_conv3.get_shape() #########
print "b_conv3:", b_conv3.get_shape() #########
print "h_conv3:", h_conv3.get_shape() #########
print "h_pool3:", h_pool3.get_shape() #########

# "In order to build a deep network, we stack several layers of this type.
# The second layer will have 64 features for each 5x5 patch."
W_conv4 = weight_variable([5, 5, 32, 64])
b_conv4 = bias_variable([64])
h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
h_pool4 = max_pool_2x2(h_conv4)

print "AFTER 4RD LAYER DECLARATION" ########
print "W_conv4:", W_conv4.get_shape() #########
print "b_conv4:", b_conv4.get_shape() #########
print "h_conv4:", h_conv4.get_shape() #########
print "h_pool4:", h_pool4.get_shape() #########

# "In order to build a deep network, we stack several layers of this type.
# The second layer will have 64 features for each 5x5 patch."
W_conv5 = weight_variable([5, 5, 64, 128])
b_conv5 = bias_variable([128])
h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)
h_pool5 = max_pool_2x2(h_conv5)

print "AFTER 5RD LAYER DECLARATION" ########
print "W_conv5:", W_conv5.get_shape() #########
print "b_conv5:", b_conv5.get_shape() #########
print "h_conv5:", h_conv5.get_shape() #########
print "h_pool5:", h_pool5.get_shape() #########


# "Now that the image size has been reduced to 7x7, we add a fully-connected layer 
# with 1024 neurons to allow processing on the entire image. We reshape the tensor
#  from the pooling layer into a batch of vectors, multiply by a weight matrix,
#  add a bias, and apply a ReLU."
W_fc1 = weight_variable([4 * 3 * 128, 512])
b_fc1 = bias_variable([512])
h_pool5_flat = tf.reshape(h_pool5, [-1, 4 * 3 * 128])
h_fc1 = tf.nn.relu(tf.matmul(h_pool5_flat, W_fc1) + b_fc1)

print "AFTER 5TH LAYER DECLARATION" ########
print "W_fc1:", W_fc1.get_shape() #########
print "b_fc1:", b_fc1.get_shape() #########
print "h_pool3_flat:", h_pool5_flat.get_shape() #########
print "h_fc1:", h_fc1.get_shape() #########


# "To reduce overfitting, we will apply dropout before the readout layer.
# We create a placeholder for the probability that a neuron's output is kept during dropout.
# This allows us to turn dropout on during training, and turn it off during testing.
# TensorFlow's tf.nn.dropout op automatically handles scaling neuron outputs in addition
# to masking them, so dropout just works without any additional scaling."
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


# Last layer declaration with softmax activation like the one layer version
W_fc2 = weight_variable([512, orlfaces.train.num_classes])
b_fc2 = bias_variable([orlfaces.train.num_classes])
logit = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

print "AFTER LAST LAYER DECLARATION" ########
print "W_fc1:", W_fc2.get_shape() #########
print "b_fc1:", b_fc2.get_shape() #########
print "y_conv:", y_conv.get_shape()


sess = tf.InteractiveSession()

# "How well does this model do? To train and evaluate it we will use code that is nearly
# identical to that for the simple one layer SoftMax network above.
# The differences are that: we will replace the steepest gradient descent optimizer with
# the more sophisticated ADAM optimizer; we will include the additional parameter
# keep_prob in feed_dict to control the dropout rate"

# cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
# cross_entropy = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y_conv, 1e-10, 1.0)))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logit, y_))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print "AFTER TRAINING DECLARATION" #######
print "cross_entropy:", cross_entropy.get_shape() #######
print "correct_prediction:", correct_prediction.get_shape() #######

sess.run((tf.initialize_all_variables()))
for i in xrange(1000):
    batch = orlfaces.train.next_batch(50)
    # print "max W vales: %g %g %g %g"%(tf.reduce_max(tf.abs(W_conv1)).eval(),tf.reduce_max(tf.abs(W_conv2)).eval(),tf.reduce_max(tf.abs(W_fc1)).eval(),tf.reduce_max(tf.abs(W_fc2)).eval())
    # print "max b vales: %g %g %g %g"%(tf.reduce_max(tf.abs(b_conv1)).eval(),tf.reduce_max(tf.abs(b_conv2)).eval(),tf.reduce_max(tf.abs(b_fc1)).eval(),tf.reduce_max(tf.abs(b_fc2)).eval())
    if i % 10 == 0:
        # train_accuracy = accuracy.eval(feed_dict = {x: batch[0], y_: batch[1], keep_prob: 1.0})
        train_accuracy = accuracy.eval(feed_dict = {x: orlfaces.train.images, y_: orlfaces.train.labels, keep_prob: 1.0})
        test_accuracy = accuracy.eval(feed_dict = {x: orlfaces.test.images, y_: orlfaces.test.labels, keep_prob: 1.0})
        print "Step %d, training accuracy %g | test accuracy %g" % (i, train_accuracy, test_accuracy)
    _, loss = sess.run([train_step, cross_entropy], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    # train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    print "loss = ", loss

print "Test accuracy %g" % accuracy.eval(feed_dict = {x: orlfaces.test.images, y_: orlfaces.test.labels, keep_prob: 1.0})
