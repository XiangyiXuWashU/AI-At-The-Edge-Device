import numpy as np
from MultiParticle import MPFormular as mpf, MPTrainModel as mpt
from MultiParticle import MPSaveDataSet as sd
import matplotlib

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt


freq = np.linspace(sd.train_startFreq, sd.train_stopFreq, mpt.n_input/2, dtype="float32")

index_batch = 0
index_dataset =  0

# Use random generate data set
data_spectrum, correct_label = sd.loadDataSetWithIndex(index_dataset)
plt.figure(figsize=(20,10))

width = 6
height = 6

for i in range(width*height):
    plt.subplot(width,height,i+1)
    plt.plot(freq, data_spectrum[index_batch+i][:int(mpt.n_input / 2)], '-b', label='B')
    plt.plot(freq, data_spectrum[index_batch+i][int(mpt.n_input / 2):mpt.n_input], '-r', label='A')
    plt.legend()

plt.tight_layout()
plt.show()