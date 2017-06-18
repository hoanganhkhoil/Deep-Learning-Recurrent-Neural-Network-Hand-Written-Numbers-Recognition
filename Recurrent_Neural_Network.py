# Author: Khoi Hoang
# Recognize Hand-written Numbers - Using Recurrent Neural Network
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn 

# Import 60,000 training examples, 10,000 testing examples from MNIST
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


# number of cycles
hm_epochs = 10

# number of classes
n_classes = 10

# Number of examples to be trained each time
batch_size = 128

# Chunk size
chunk_size = 28

# Number of chunks
n_chunks = 28

rnn_size = 128

x = tf.placeholder('float',[None, n_chunks, chunk_size])
y = tf.placeholder('float')


def recurrent_neural_network(x):
    layer = {'weights': tf.Variable(tf.random_normal([rnn_size, n_classes])),
            'biases': tf.Variable(tf.random_normal([n_classes]))}

    # Reshape the input to fit RNN model
    x = tf.transpose(x,[1,0,2])
    x = tf.reshape(x,[-1,chunk_size])
    x = tf.split(x,n_chunks,0)

    lstm_cell = rnn.BasicLSTMCell(rnn_size)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # output = final output -> outputs[-1]
    output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']

    return output



def train_neural_network():
    # x in this case is not called yet.
    # The function is called, but not actually run yet.
    # Since we haven't run the session.
    # All of this is just to set up a model for the cost and optimizer to follow.
    prediction = recurrent_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)




    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Training data
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                # Extract examples from 'mnist'.
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)

                # Reshape x
                epoch_x = epoch_x.reshape((batch_size,n_chunks,chunk_size))

                # Now we actually run the session -> run optimizer, cost, and prediction model
                # x now will actually be passed to recurrent_neural_network(x)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            print 'Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss


        # tf.argmax return the index of the maximum in the labels [001000000] ->return index 2
        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))

        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print 'Accuracy:', accuracy.eval({x:mnist.test.images.reshape((-1,n_chunks,chunk_size)), y:mnist.test.labels})


train_neural_network()
        
