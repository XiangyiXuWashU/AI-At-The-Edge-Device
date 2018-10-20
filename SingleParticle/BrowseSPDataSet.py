import numpy as np
from SingleParticle import SPFormular as ms, SPTrainModel as msm
from SingleParticle import SPSaveDataSet as sd
import matplotlib

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

freq = np.linspace(sd.train_startFreq, sd.train_stopFreq, msm.n_input/2, dtype="float32")

index_batch = 0
index_dataset =  0

# Use random generate data set
data_spectrum, correct_label = sd.loadDataSetWithIndex(index_dataset)
plt.figure(figsize=(20,10))

width = 6
height = 6

for i in range(width*height):
    plt.subplot(width,height,i+1)
    plt.plot(freq, data_spectrum[index_batch+i][:int(msm.n_input / 2)], '-b', label='B')
    plt.plot(freq, data_spectrum[index_batch+i][int(msm.n_input / 2):msm.n_input], '-r', label='A')
    plt.legend()

plt.tight_layout()
plt.show()