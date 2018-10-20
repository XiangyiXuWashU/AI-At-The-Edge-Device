import tensorflow as tf
from Shift import ModeShift as ms
import numpy as np


# Network Parameters
n_input = 300    # number of neurons for input layer
n_hidden_1 = 200 # 1st hidden layer number of neurons
n_hidden_2 = 100 # 2nd hidden layer number of neurons
n_hidden_3 = 50 # 3nd hidden layer number of neurons
n_output = 1     # number of neurons for output layer


class ModeShiftTrainNet:
    def __init__(self):
        self.weights = {
            'w1': tf.Variable(tf.random_normal([n_input, n_hidden_1],dtype=tf.float32)),
            'w2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2],dtype=tf.float32)),
            'w3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3], dtype=tf.float32)),
            'out': tf.Variable(tf.random_normal([n_hidden_3, n_output],dtype=tf.float32))
        }
        self.biases = {
            'b1': tf.Variable(tf.random_normal([n_hidden_1],dtype=tf.float32)),
            'b2': tf.Variable(tf.random_normal([n_hidden_2],dtype=tf.float32)),
            'b3': tf.Variable(tf.random_normal([n_hidden_3], dtype=tf.float32)),
            'out': tf.Variable(tf.random_normal([n_output],dtype=tf.float32))
        }

    # predict mode shift value
    def predict(self, x):
        # Hidden fully connected layer with 100 neurons
        layer_1 = tf.add(tf.matmul(x, self.weights['w1']), self.biases['b1'])
        # Hidden fully connected layer with 100 neurons
        layer_2 = tf.add(tf.matmul(tf.sigmoid(layer_1), self.weights['w2']), self.biases['b2'])
        # Hidden fully connected layer with 100 neurons
        layer_3 = tf.add(tf.matmul(tf.sigmoid(layer_2), self.weights['w3']), self.biases['b3'])
        # Output fully connected layer with a neuron for output result
        result = tf.add(tf.matmul(tf.sigmoid(layer_3), self.weights['out']), self.biases['out'])

        return result

    # Calculate loss
    def evaluate(self, input, deltaT):
        # Predict shift
        result = self.predict(input)

        # Calculate Loss and Error
        loss = result-deltaT
        error = tf.abs(result-deltaT)

        return [result,loss,error]


if __name__ == '__main__':

    # Define a model instance
    net = ModeShiftTrainNet()
    net.__init__()

    # Assume a delta
    delta = 0.5

    # Generate spectrum
    resonance = ms.ModeShift()
    resonance.__init__(k0=20e+6, k1=20e+6)
    _, _, spectrumAfter = resonance.calculateSpectrum(
        startFreq=-1, stopFreq=1, number=n_input, deltaT=delta)
    # Convert numpy array to tensor
    data_np = np.asarray([spectrumAfter], np.float32)
    X = tf.convert_to_tensor(data_np, dtype=tf.float32)


    # Calculate loss
    _, loss, error = net.evaluate(X, delta)

    # Initializing the variables
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        print (sess.run(error))

