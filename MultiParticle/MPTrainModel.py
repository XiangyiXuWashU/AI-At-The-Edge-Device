import tensorflow as tf
import matplotlib
import math
matplotlib.use('TkAgg')
from MultiParticle import MPFormular as mpf
import numpy as np


# Network Parameters
n_input = 600    # number of neurons for input layer
n_hidden_1 = 200 # 1st hidden layer number of neurons 200
n_hidden_2 = 200 # 2nd hidden layer number of neurons 200
n_hidden_3 = 100  # 3nd hidden layer number of neurons 100
# n_hidden_4 = 50  # 3nd hidden layer number of neurons 100
n_output = 2     # number of neurons for output layer


class MPTrainNet:
    def __init__(self):
        with tf.name_scope('weights'):
            self.weights = {
                'w1': tf.Variable(tf.random_normal([n_input, n_hidden_1],dtype=tf.float32), name='weight1'),
                'w2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2],dtype=tf.float32), name='weight2'),
                'w3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3], dtype=tf.float32), name='weight3'),
                # 'w4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4], dtype=tf.float32), name='weight4'),
                'out': tf.Variable(tf.random_normal([n_hidden_3, n_output],dtype=tf.float32), name='weight_out')
            }
        with tf.name_scope('biases'):
            self.biases = {
                'b1': tf.Variable(tf.random_normal([n_hidden_1],dtype=tf.float32), name='bias1'),
                'b2': tf.Variable(tf.random_normal([n_hidden_2],dtype=tf.float32), name='bias2'),
                'b3': tf.Variable(tf.random_normal([n_hidden_3], dtype=tf.float32), name='bias3'),
                # 'b4': tf.Variable(tf.random_normal([n_hidden_4], dtype=tf.float32), name='bias4'),
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

        # Hidden fully connected layer4
        # with tf.name_scope('layer_3'):
        #     layer_4 = tf.add(tf.matmul(tf.nn.sigmoid(layer_3), self.weights['w4']), self.biases['b4'])

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
    net = MPTrainNet()
    net.__init__()

    # Generate spectrum
    resonance = mpf.MultiParticle()
    resonance.__init__(ep = 1.59**2, em = 1, ec = 1.45**2, vc = 1.47e-16)

    npStackX = []
    npStackY = []
    times = 3

    for _ in range(times):

        R1 = 100 / mpf.scale_R  # Real R = scale_R*R*1e-9
        R2 = 80 / mpf.scale_R  # Real R = scale_R*R*1e-9
        fr1 = 200 / mpf.scale_FR  # Real fr = scale_FR*fr*1e-3
        fr2 = 200 / mpf.scale_FR  # Real fr = scale_FR*fr*1e-3
        k0 = 20 / mpf.scale_K0  # Real k0 = scale_K0*k0*1e+6
        k1 = 20 / mpf.scale_K1  # Real k1 = scale_K1*k1*1e+6
        kx2 = 0.5 * math.pi / mpf.scale_KX  # Real kx = scale_KX*kX


        freq, spectrumBefore = resonance.calculateSpectrum(R=[R1],kx=[0],fr=[fr1],k0=k0, k1=k1,
                                                           startFreq=-5,stopFreq=5,
                                                           number=n_input/2, wavelength=1550e-9)

        _, spectrumAfter = resonance.calculateSpectrum(R=[R1, R2],kx=[0, kx2],fr=[fr1, fr2],k0=k0, k1=k1,
                                                           startFreq=-5,stopFreq=5,
                                                           number=n_input/2, wavelength=1550e-9)

        # Convert numpy array to tensor
        data_np = np.asarray([np.concatenate((spectrumBefore, spectrumAfter), axis=0)], np.float32)
        npStackX = np.append(npStackX, data_np)

        correct_label = np.asarray([[R1, R2, fr1, fr2, kx2, k0, k1]], np.float32)
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
