import numpy as np
from sklearn.model_selection import train_test_split

# To be compatible with python3
try:
    import cPickle as pickle
except ImportError:
    import pickle

import tensorflow as tf


def convert_to_one_hot(vector, num_classes=None):
    result = np.zeros((len(vector), num_classes), dtype='float32')
    result[np.arange(len(vector)), vector] = 1
    return result

# parameters
learning_rate = 0.05
training_epochs = 1500
batch_size = 100
display_step = 100


with open('./mfcc_features.pkl', 'rb') as f:
    X = pickle.load(f) # use encoding='latin1' for python3 compatibility
with open('./genre_targets.pkl', 'rb') as f:
    y = pickle.load(f)


labels = list(np.unique(y))
categorical_to_numercial = []
for i in range(len(y)):
    categorical_to_numercial.append(labels.index(y[i]))

Y = convert_to_one_hot(categorical_to_numercial, 10)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

# tf graph Input
x = tf.placeholder(tf.float32, shape=[None, 13], name='x_data')
y_true = tf.placeholder(tf.float32, shape=[None, 10], name='y_data')

# We start with random weights a
W = tf.Variable(tf.random_normal([13, 10], stddev=1e-4))
b = tf.Variable(tf.zeros([10]))

# construct model
pred = tf.nn.softmax(tf.matmul(x, W) + b)  # Softmax

# minimize error using cross entropy
loss = tf.reduce_mean(-tf.reduce_sum(y_true * tf.log(tf.clip_by_value(pred, 1e-10, 1.0)), reduction_indices=1))
#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y_true))

# gradient descent
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# initializing the variables
init = tf.global_variables_initializer()

tf.summary.FileWriter("/home/michael/Documents/mgc/mlp_tensorflow/", tf.get_default_graph()).close()

# launch the graph
with tf.Session() as sess:
    sess.run(init)

    # training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(X_train.shape[0] / batch_size)

        # loop over all batches
        for i in range(total_batch):
            #idx = np.random.permutation(X_train.shape[0])[0:128]  # Easy minibatch of size 64
            beg = 0
            end = batch_size
            # run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, loss], feed_dict={x: X_train[beg:end],
                                                                         y_true: y_train[beg:end]})
            # compute average loss
            avg_cost += c / total_batch
            beg = beg + end
            end = end + beg

            if end > X_train.shape[0]:
                end = X_train.shape[0]

        # display logs per epoch step
        if (epoch + 1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "loss=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y_true, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: X_test, y_true: y_test}))

