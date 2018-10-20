import tensorflow as tf
import numpy as np
import time
import math
import os
from MultiParticle import MPFormular as mpf, MPTrainModel as mpt
from MultiParticle import MPSaveDataSet as sd
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

# Parameters for training spectrum
r_min = 20/mpf.scale_R              #Real R = scale_R*R*1e-9
r_max = 100/mpf.scale_R             #Real R = scale_R*R*1e-9
fr_min = 40/mpf.scale_FR            #Real fr = scale_FR*fr*1e-3
fr_max = 400/mpf.scale_FR           #Real fr = scale_FR*fr*1e-3
k0_min = 4/mpf.scale_K0             #Real k0 = scale_K0*k0*1e+6
k0_max = 40/mpf.scale_K0            #Real k0 = scale_K0*k0*1e+6
k1_min = 0.8/mpf.scale_K1           #Real k1 = scale_K1*k1*1e+6
k1_max = 200/mpf.scale_K1           #Real k1 = scale_K1*k1*1e+6
kx_min = 0.0                        #Real kx = scale_KX*kX
kx_max = 1.0*math.pi/mpf.scale_KX   #Real kx = scale_KX*kX

train_ep = 1.59**2
train_em = 1.0
train_ec = 1.45**2
train_vc = 1.47e-16
train_startFreq = -10.0
train_stopFreq = 10.0

#Set numpy array display precision
np.set_printoptions(precision=3)


# Figure Path
figurePath = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) \
           + '/Share/' \
           + 'TestMPSensing.png'

# Generate testing batches
def generateTestBatches():
    #Using generated dataset
    data_spectrum, correct_label = sd.loadDataSet()

    TensorX = tf.convert_to_tensor(data_spectrum, dtype=tf.float32)
    TensorY = tf.convert_to_tensor(correct_label, dtype=tf.float32)

    return [TensorX, TensorY]

def generateSingleBatch():

    data_spectrum, correct_label = sd.generateTrainBatches(batch_size=1)

    TensorX = tf.convert_to_tensor(data_spectrum, dtype=tf.float32)
    TensorY = tf.convert_to_tensor(correct_label, dtype=tf.float32)

    return [TensorX, TensorY]

# Load the multi nano-particle sensing model
def loadMPModel():
    # Construct model
    trainNet = mpt.MPTrainNet()
    trainNet.__init__()

    # create a saver
    saver = tf.train.Saver()

    # Model Name
    modelName = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) \
                + '/MPModel/' \
                + 'MPModel' \
                + '.ckpt'

    sess = tf.Session()

    try:
        # Initialize variables
        sess.run(tf.global_variables_initializer())
        # Try to restore the trained session
        saver.restore(sess, modelName)

    except:
        print("Load model fail!")

    return [sess, trainNet]

def closeMPModel():
    tf.reset_default_graph()

# Just test one frame
def testSingleMPModel(sess, trainNet):
    TensorX, TensorY = generateSingleBatch()
    predict, error, loss = trainNet.evaluate(TensorX, TensorY)

    # Radius prediction
    predictR1 = tf.cast(predict[:1, 0], tf.float32)
    Predict_R1 = mpf.scale_R * predictR1.eval(session=sess)
    Label_R1 = mpf.scale_R * TensorY[:1, 0].eval(session=sess)
    Error_R1 = np.abs(Predict_R1-Label_R1)
    Precision_R1 = np.subtract(1, Error_R1/Label_R1)*100

    predictR2 = tf.cast(predict[:1, 1], tf.float32)
    Predict_R2 = mpf.scale_R * predictR2.eval(session=sess)
    Label_R2 = mpf.scale_R * TensorY[:1, 1].eval(session=sess)
    Error_R2 = np.abs(Predict_R2-Label_R2)
    Precision_R2 = np.subtract(1, Error_R2/Label_R2)*100


    # Edit message send to iPhone
    message = 'MPSensing:\n' \
              + "Label_R1  = %6.2fnm  " % Label_R1 \
              + "Predict_R1  = %6.2fnm  " % Predict_R1 \
              + "Error_R1 = %6.2fnm  " % Error_R1 \
              + "Precision = %4.1f%%\n" % Precision_R1 \
              + "Label_R2  = %6.2fnm  " % Label_R2 \
              + "Predict_R2  = %6.2fnm  " % Predict_R2 \
              + "Error_R2 = %6.2fnm  " % Error_R2 \
              + "Precision = %4.1f%%\n" % Precision_R2 \
              + "StartFreq = %6.2e\n" % train_startFreq \
              + "StopFreq = %6.2e\n" % train_stopFreq \
              + "Number = %6.0e\n" % int(mpt.n_input/2) \
              + "Spectrum = %s\n" % TensorX.eval(session=sess)[0] \
              + "SpectrumEnd"\

    return  message.encode()


# Test one batch
def testBatchMPModel(sess, trainNet, batch_size = 200):

    TensorX, TensorY = generateTestBatches()
    predict, error, loss = trainNet.evaluate(TensorX, TensorY)

    # Plot R1 prediction
    plt.figure()

    plt.subplot(2,1,1)
    plt.plot(mpf.scale_R * TensorY[:batch_size, 0].eval(session=sess), mpf.scale_R* sess.run(predict)[:, 0], '.',
             color='red', label='R1')
    plt.xlabel('Assume Radius(nm)')
    plt.ylabel('Predict Radius(nm)')
    plt.legend()

    plt.subplot(2,1,2)
    plt.plot(mpf.scale_R * TensorY[:batch_size, 1].eval(session=sess), mpf.scale_R* sess.run(predict)[:, 1], '.',
             color='red', label='R2')
    plt.xlabel('Assume Radius(nm)')
    plt.ylabel('Predict Radius(nm)')
    plt.legend()

    plt.tight_layout()

    # Save the figure
    plt.savefig(figurePath, bbox_inches='tight')
    plt.close()

    message = b'BatchMultiParticleSensing:'

    return message

if __name__ == '__main__':

    sess, trainNet = loadMPModel()
    # testSingleMPModel(sess=sess, trainNet=trainNet)
    testBatchMPModel(sess=sess, trainNet=trainNet, batch_size=sd.size)



