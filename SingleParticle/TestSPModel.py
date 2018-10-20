import tensorflow as tf
import numpy as np
import time
import os
from SingleParticle import SPFormular as ms, SPTrainModel as msm
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt


# Parameters for training spectrum
r_min = 20/ms.scale_R       #Real R = scale_R*R*1e-9
r_max = 200/ms.scale_R      #Real R = scale_R*R*1e-9
fr_min = 40/ms.scale_FR     #Real fr = scale_FR*fr*1e-3
fr_max = 400/ms.scale_FR    #Real fr = scale_FR*fr*1e-3
k0_min = 4/ms.scale_K0      #Real k0 = scale_K0*k0*1e+6
k0_max = 40/ms.scale_K0     #Real k0 = scale_K0*k0*1e+6
k1_min = 0.8/ms.scale_K1    #Real k1 = scale_K1*k1*1e+6
k1_max = 200/ms.scale_K1    #Real k1 = scale_K1*k1*1e+6
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
           + 'TestSPSensing.png'

# Generate testing batches
def generateTestBatches(batch_size = 200):
    # Generate spectrum
    resonance = ms.SingleParticle()
    resonance.__init__(ep = train_ep, em = train_em, ec = train_ec, vc = train_vc)

    npStackX = []
    npStackY = []

    #valueNumber is the number of generate spectrum which has split
    valueNumber = 0

    while valueNumber < batch_size:
        setR = np.random.uniform(r_min, r_max)
        fr = np.random.uniform(fr_min, fr_max)
        k0 = np.random.uniform(k0_min, k0_max)
        k1 = np.random.uniform(k1_min, k1_max)

        freq, spectrumBefore, spectrumAfter = resonance.calculateSpectrum(
            startFreq=train_startFreq, stopFreq=train_stopFreq, number=msm.n_input/2, R=setR, fr=fr, k0 = k0, k1 = k1)

        # Judge whether the spectrum has split
        if resonance.findValley(spectrumAfter) == 2:
            valueNumber = valueNumber + 1
            # Convert numpy array to tensor
            data_np = np.asarray([resonance.normalize(np.concatenate((spectrumBefore, spectrumAfter), axis=0))],
                                 np.float32)
            npStackX = np.append(npStackX, data_np)

            correct_label = np.asarray([[setR, fr, k0, k1]], np.float32)
            npStackY = np.append(npStackY, correct_label)

    TensorX = tf.convert_to_tensor(npStackX.reshape(batch_size, msm.n_input), dtype=tf.float32)
    TensorY = tf.convert_to_tensor(npStackY.reshape(batch_size, msm.n_output), dtype=tf.float32)

    return [TensorX, TensorY]

# Load the single nano-particle sensing model
def loadSPModel():
    # Construct model
    trainNet = msm.SPTrainNet()
    trainNet.__init__()

    # create a saver
    saver = tf.train.Saver()

    # Model Name
    modelName = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) \
                + '/SPModel/' \
                + 'SPModel' \
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

def closeSPModel():
    tf.reset_default_graph()


# Just test one frame
def testSingleSPModel(sess, trainNet):
    TensorX, TensorY = generateTestBatches(batch_size=1)
    predict, error, loss = trainNet.evaluate(TensorX, TensorY)

    # Radius prediction
    predictR = tf.cast(predict[:1, 0], tf.float32)
    Predict_R = ms.scale_R * predictR.eval(session=sess)
    Label_R = ms.scale_R * TensorY[:1, 0].eval(session=sess)
    Error_R = np.abs(Predict_R-Label_R)
    Precision_R = np.subtract(1, Error_R/Label_R)*100

    # FR prediction
    predictFR = tf.cast(predict[:1, 1], tf.float32)
    Predict_FR = ms.scale_FR*1e-3 * predictFR.eval(session=sess)
    Label_FR = ms.scale_FR*1e-3 * TensorY[:1, 1].eval(session=sess)
    Error_FR = np.abs(Predict_FR-Label_FR)
    Precision_FR = np.subtract(1, Error_FR/Label_FR)*100

    # K0 prediction
    predictK0 = tf.cast(predict[:1, 2], tf.float32)
    Predict_K0 = ms.scale_K0*1e+6 * predictK0.eval(session=sess)
    Label_K0 = ms.scale_K0*1e+6 * TensorY[:1, 2].eval(session=sess)
    Error_K0 = np.abs(Predict_K0-Label_K0)
    Precision_K0 = np.subtract(1, Error_K0/Label_K0)*100

    # K1 prediction
    predictK1 = tf.cast(predict[:1, 3], tf.float32)
    Predict_K1 = ms.scale_K1*1e+6 * predictK1.eval(session=sess)
    Label_K1 = ms.scale_K1 *1e+6* TensorY[:1, 3].eval(session=sess)
    Error_K1 = np.abs(Predict_K1-Label_K1)
    Precision_K1 = np.subtract(1, Error_K1/Label_K1)*100
    #SingleNano-particleSensing
    # Edit message send to iPhone
    message = 'SPSensing:\n' \
              + "Label_R0  = %6.2fnm  " % Label_R \
              + "Predict_R0  = %6.2fnm  " % Predict_R \
              + "Error_R0 = %6.2fnm  " % Error_R \
              + "Precision = %4.1f%%\n" % Precision_R \
              + "Label_FR  = %6.2e  " % Label_FR \
              + "Predict_FR  = %6.2e  " % Predict_FR \
              + "Error_FR = %6.2e  " % Error_FR \
              + "Precision = %4.1f%%\n" % Precision_FR \
              + "Label_K0  = %6.2e  " % Label_K0 \
              + "Predict_K0  = %6.2e  " % Predict_K0 \
              + "Error_K0 = %6.2e  " % Error_K0 \
              + "Precision = %4.1f%%\n" % Precision_K0 \
              + "Label_K1  = %6.2e  " % Label_K1 \
              + "Predict_K1  = %6.2e  " % Predict_K1 \
              + "Error_K1 = %6.2e  " % Error_K1 \
              + "Precision = %4.1f%%\n" % Precision_K1 \
              + "StartFreq = %6.2e\n" % train_startFreq \
              + "StopFreq = %6.2e\n" % train_stopFreq \
              + "Number = %6.0e\n" % int(msm.n_input/2) \
              + "Spectrum = %s\n" % TensorX.eval(session=sess)[0] \
              + "SpectrumEnd"\

    return  message.encode()


# Test one batch
def testBatchSPModel(sess, trainNet, batch_size = 200):
    TensorX, TensorY = generateTestBatches(batch_size=batch_size)
    predict, error, loss = trainNet.evaluate(TensorX, TensorY)

    # Plot R prediction
    plt.figure()

    plt.subplot(221)
    plt.plot(ms.scale_R * TensorY[:batch_size, 0].eval(session=sess), ms.scale_R* sess.run(predict)[:, 0], '.',
             color='red', label='R')
    plt.xlabel('Assume Radius(nm)')
    plt.ylabel('Predict Radius(nm)')
    plt.legend()

    # Plot FR prediction
    plt.subplot(222)
    plt.plot(ms.scale_FR*1e-3 * TensorY[:batch_size, 1].eval(session=sess), ms.scale_FR*1e-3*sess.run(predict)[:, 1], '.',
             color='green', label='FR')
    plt.xlabel('Assume FR')
    plt.ylabel('Predict FR')
    plt.legend()

    # Plot K0 prediction
    plt.subplot(223)
    plt.plot(ms.scale_K0*1e+6 * TensorY[:batch_size, 2].eval(session=sess), ms.scale_K0*1e+6 * sess.run(predict)[:, 2], '.',
             color='orange', label='K0')
    plt.xlabel('Assume K0')
    plt.ylabel('Predict K0')
    plt.legend()

    # Plot K1 prediction
    plt.subplot(224)
    plt.plot(ms.scale_K1*1e+6 * TensorY[:batch_size, 3].eval(session=sess), ms.scale_K1*1e+6 * sess.run(predict)[:, 3], '.',
             color='blue', label='K1')
    plt.xlabel('Assume K1')
    plt.ylabel('Predict K1')
    plt.legend()

    plt.tight_layout()

    # Save the figure
    plt.savefig(figurePath, bbox_inches='tight')
    plt.close()

    message = b'BatchSingleParticleSensing:'

    return message

if __name__ == '__main__':

    sess, trainNet = loadSPModel()
    # testSingleSPModel(sess=sess, trainNet=trainNet)
    testBatchSPModel(sess=sess, trainNet=trainNet, batch_size=200)



