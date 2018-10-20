import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')
from SingleParticle import SPFormular as ms
import numpy as np


# Network Parameters
n_input = 600    # number of neurons for input layer
n_hidden_1 = 200 # 1st hidden layer number of neurons 300
n_hidden_2 = 100 # 2nd hidden layer number of neurons 200
n_hidden_3 = 50  # 3nd hidden layer number of neurons 100
n_output = 4     # number of neurons for output layer


class SPTrainNet:
    def __init__(self):
        with tf.name_scope('weights'):
            self.weights = {
                'w1': tf.Variable(tf.random_normal([n_input, n_hidden_1],dtype=tf.float32), name='weight1'),
                'w2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2],dtype=tf.float32), name='weight2'),
                'w3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3], dtype=tf.float32), name='weight3'),
                'out': tf.Variable(tf.random_normal([n_hidden_3, n_output],dtype=tf.float32), name='weight_out')
            }
        with tf.name_scope('biases'):
            self.biases = {
                'b1': tf.Variable(tf.random_normal([n_hidden_1],dtype=tf.float32), name='bias1'),
                'b2': tf.Variable(tf.random_normal([n_hidden_2],dtype=tf.float32), name='bias2'),
                'b3': tf.Variable(tf.random_normal([n_hidden_3], dtype=tf.float32), name='bias3'),
                'out': tf.Variable(tf.random_normal([n_output],dtype=tf.float32), name='bias_out')
            }

    # predict mode split value
    def predict(self, x):

        # Hidden fully connected layer1
        with tf.name_scope('layer_1'):
            layer_1 = tf.add(tf.matmul(x, self.weights['w1']), self.biases['b1'])

        # Hidden fully connected layer2
        with tf.name_scope('layer_2'):
            layer_2 = tf.add(tf.matmul(tf.nn.relu(layer_1), self.weights['w2']), self.biases['b2'])

        # Hidden fully connected layer3
        with tf.name_scope('layer_3'):
            layer_3 = tf.add(tf.matmul(tf.nn.sigmoid(layer_2), self.weights['w3']), self.biases['b3'])

        # Output fully connected layer with a neuron for output result
        with tf.name_scope('result'):
            result = tf.add(tf.matmul(tf.nn.relu(layer_3), self.weights['out']), self.biases['out'])

        return result

    # Calculate loss
    def evaluate(self, input, CorrectLabel):
        # Predict diameter
        result = self.predict(input)

        # Calculate Loss and Error
        error = tf.abs(result - CorrectLabel)
        loss = tf.square(result - CorrectLabel)

        return [result,error,loss]


if __name__ == '__main__':

    # Define a model instance
    net = SPTrainNet()
    net.__init__()

    # Generate spectrum
    resonance = ms.SingleParticle()
    resonance.__init__(ep = 1.59**2, em = 1, ec = 1.45**2, vc = 1.47e-16)

    npStackX = []
    npStackY = []
    times = 3

    for _ in range(times):

        R = 150.0
        fr = 200.0
        k0 = 200
        k1 = 200

        _, spectrumBefore, spectrumAfter = resonance.calculateSpectrum(
            startFreq = -5, stopFreq = 5, number =n_input/2, R = R, fr =fr, k0 = 200, k1 = 200)

        # Convert numpy array to tensor
        data_np = np.asarray([resonance.normalize(np.concatenate((spectrumBefore, spectrumAfter), axis=0))],
                             np.float32)
        npStackX = np.append(npStackX, data_np)

        correct_label = np.asarray([[R, fr, k0, k1]], np.float32)
        npStackY = np.append(npStackY, correct_label)


    X = tf.convert_to_tensor(npStackX.reshape(times,n_input), dtype=tf.float32)
    Y = tf.convert_to_tensor(npStackY.reshape(times,n_output), dtype=tf.float32)

    # Initializing the variables
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        result, error, loss = net.evaluate(X, Y)
        loss_op = tf.reduce_mean(result)

        # print ("Y Label \n %s\n"% sess.run(Y))
        # print ("Predict \n %s\n"% sess.run(result))
        print ("Error \n%s\n"% sess.run(error))
        print("Error \n%s\n" % sess.run(tf.reduce_mean(error, 0)[1]))

        print ("Loss \n%s\n"% sess.run(loss))
        # print ("Loss operator \n%s"% sess.run(loss_op))
