import tensorflow as tf
import numpy as np
import os
from Shift import ModeShift as ms, ModeShiftTrainModel as msm
from matplotlib import pyplot as plt

# Parameters for training spectrum
shift_range = 1.8
train_k0 = 20e+6
train_k1 = 20e+6
train_startFreq = -2
train_stopFreq = 2


# Inference from trained model
def predictFromModel(deltaT, k0=20e+6, k1=20e+6, startFreq=-1, stopFreq=1):

    # Generate spectrum
    resonance = ms.ModeShift()
    resonance.__init__(k0=k0, k1=k1)
    _, _, spectrumAfter = resonance.calculateSpectrum(
        startFreq=startFreq, stopFreq=stopFreq, number=msm.n_input, deltaT=deltaT)
    # Convert numpy array to tensor
    data_np = np.asarray([spectrumAfter], np.float32)
    X = tf.convert_to_tensor(data_np, dtype=tf.float32)

    # Calculate loss
    result, loss, error = trainNet.evaluate(X, deltaT)

    return  [result[0][0], loss, error]

if __name__ == '__main__':

    # Construct model
    trainNet = msm.ModeShiftTrainNet()
    trainNet.__init__()

    # create a saver
    saver = tf.train.Saver()

    # Initializing the variables
    init = tf.global_variables_initializer()

    modelName = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) \
                + '/ShiftModel/' \
                + 'shiftModel' \
                + '_k0:%.1e' % train_k0 \
                + '_k1:%.1e' % train_k1 \
                + '_start:%.1f' % train_startFreq \
                + '_stop:%.1f' % train_stopFreq \
                + '_number:%.0f' % msm.n_input \
                + '.ckpt'


    with tf.Session() as sess:

        sess.run(init)

        # Restore model successfully

        restoreFlag = 1

        # Try to restore the trained session
        try:
            saver.restore(sess, modelName)

        except:
            restoreFlag = 0
            print("Ｌoad model fail！")

        if restoreFlag:

            # Generate mode shift array and predict array
            setDeltaT = np.linspace(-shift_range, shift_range, 100, dtype="float32")
            predictT  = np.empty(shape=[len(setDeltaT), 0])

            #　Inference for each setDelatT
            for i in range(len(setDeltaT)):

                result, loss, error = predictFromModel(setDeltaT[i],k0=train_k0, k1=train_k1,
                                                       startFreq=train_startFreq, stopFreq=train_stopFreq)
                predictT = np.append(predictT, result.eval(session=sess))

                # print("DeltaT:%.6f" % setDeltaT[i]," Predict:%.6f" % sess.run(result),
                #       " Error:%.6f" % sess.run(error))

            print(sess.run(trainNet.weights['w1'][0]))

            # Plot the testing figure
            plt.plot(setDeltaT, predictT, '.', color='red',label='Test Mode Shift')
            plt.xlabel('Assume Shift(GHz)')
            plt.ylabel('Predict Shift(GHz)')
            plt.legend()
            plt.show()

