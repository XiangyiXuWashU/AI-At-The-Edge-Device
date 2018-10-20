import numpy as np
from matplotlib import pyplot as plt

class ModeShift:
    def __init__(self, k0 = 20e+6, k1 = 20e+6):
        self.k0 = k0
        self.k1 = k1

    # Default start frequency is -1GHz; stop frequency is 1GHz; number is 1000
    # deltaT = KT*DeltaT, default value is 0.5 GHz

    def calculateSpectrum(self, startFreq = -1, stopFreq = 1, number =1000, deltaT = 0.5):
        # freq is x-Axis, spectrumBefore and spectrumAfter are y-Axis
        freq = np.linspace(startFreq*1e+9, stopFreq*1e+9, number,dtype="float32")
        spectrumBefore = (freq ** 2 + 0.25 * ((self.k0 - self.k1) ** 2)) /\
                         (freq ** 2 + 0.25 * ((self.k0 + self.k1) ** 2))
        spectrumAfter = ((freq + deltaT*1e+9) ** 2 + 0.25 * ((self.k0 - self.k1) ** 2)) /\
                        ((freq + deltaT*1e+9) ** 2 + 0.25 * ((self.k0 + self.k1) ** 2))

        return [freq, spectrumBefore, spectrumAfter]



if __name__ == '__main__':
    resonance = ModeShift()
    resonance.__init__(k0 = 20e+6, k1 = 20e+6)
    freq, spectrumBefore, spectrumAfter = resonance.calculateSpectrum(
        startFreq = -1, stopFreq = 1, number =1000, deltaT = 0.5)

    plt.plot(freq,spectrumBefore,'-r',label='Spectrum Before')
    plt.plot(freq,spectrumAfter, '-b',label='Spectrum After')
    plt.xlabel('Frequency(GHz)')
    plt.ylabel('Normalized intensity')
    plt.legend()
    plt.show()



