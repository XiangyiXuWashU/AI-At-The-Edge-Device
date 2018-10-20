from SingleParticle import SPFormular as ms, SPTrainModel as msm
import numpy as np
import os
import time

#Batch_size
size = 200

# Generate DataSet Number
dataset_number = 2000

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


# Create a ModeSplit instance
resonance = ms.SingleParticle()
resonance.__init__(ep = train_ep, em = train_em, ec = train_ec, vc = train_vc)

# Generate training batches for each sess.run
def generateTrainBatches(batch_size = size):
    npStackX = []
    npStackY = []

    #valueNumber is the number of generate spectrum which has split
    valueNumber = 0

    while valueNumber < batch_size:
        # Assume a diameter of particle(nm)
        # Assume a fr[40*e-3 - 400*e-3]
        # Assume a k0[40*e+5 - 400*e+5]
        # Assume a k1[40*e+5 - 400*e+5]
        R = np.random.uniform(r_min, r_max)
        fr = np.random.uniform(fr_min, fr_max)
        k0 = np.random.uniform(k0_min, k0_max)
        k1 = np.random.uniform(k1_min, k1_max)

        freq, spectrumBefore, spectrumAfter = resonance.calculateSpectrum(
            startFreq=train_startFreq, stopFreq=train_stopFreq, number=msm.n_input/2, R=R, fr=fr, k0 = k0, k1 = k1)

        # Judge whether the spectrum has split
        if resonance.findValley(spectrumAfter) == 2:
            valueNumber = valueNumber + 1
            # Convert numpy array to tensor
            data_np = np.asarray([resonance.normalize(np.concatenate((spectrumBefore, spectrumAfter), axis=0))],
                                 np.float32)
            npStackX = np.append(npStackX, data_np)

            correct_label = np.asarray([[R, fr, k0, k1]], np.float32)
            npStackY = np.append(npStackY, correct_label)

    return [npStackX.reshape(batch_size, msm.n_input), npStackY.reshape(batch_size, msm.n_output)]


def loadDataSet():
    dataSetName = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) \
                  + '/SPDataSet/' \
                  + '%d' %  np.random.randint(0,dataset_number-1)\
                  + '.npz'
    data = np.load(dataSetName)

    return  [data['name1'], data['name2']]

def loadDataSetWithIndex(index):
    dataSetName = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) \
                  + '/SPDataSet/' \
                  + '%d' %  index\
                  + '.npz'
    data = np.load(dataSetName)

    return  [data['name1'], data['name2']]

if __name__ == '__main__':

    # Delete all the original file
    dirPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) \
              + '/SPDataSet/'

    fileList = os.listdir(dirPath)
    for fileName in fileList:
        os.remove(dirPath + "/" + fileName)

    for i in range(dataset_number):

        data_spectrum, correct_label = generateTrainBatches(batch_size = size)

        dataSetName = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) \
                    + '/SPDataSet/' \
                    + '%d'%i \
                    + '.npz'

        np.savez(dataSetName, name1=data_spectrum, name2=correct_label)
        if i % 10 == 0:
            print("Have Generated %d" %i)

    print("Finished!")

    # data1, data2 = loadDataSet()
    # print(data1)
    # print(data2)



