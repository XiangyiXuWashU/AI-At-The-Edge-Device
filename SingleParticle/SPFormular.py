import numpy as np
import math
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

# light speed
c = 3e8

# Scale R for training
scale_R  = 0.1
scale_FR = 1.0
scale_K0 = 0.1
scale_K1 = 0.5

class SingleParticle:
    def __init__(self, ep = 1.59**2, em = 1, ec = 1.45**2, vc = 1.47e-16):
        self.ep = ep
        self.em = em
        self.ec = ec
        self.vc = vc

    # R is radius of particle(nm)
    # fr gives the information of the position of attached particle

    def calculateSpectrum(self, startFreq = -5.0, stopFreq = 5.0, number =1000,
                          R = 200.0, fr =100.0, k0 = 200.0, k1 = 40.0, wavelength = 1550e-9):
        # Convert radius from nm to m
        # 0.5 is for training balance
        RADIUS = scale_R*R*1e-9

        # Scale fr
        FR = scale_FR*fr*1e-3
        
        # Convert Kapa0 and Kapa1
        # scale is for training balance
        Kapa0 = scale_K0*k0*1e+6
        Kapa1 = scale_K1*k1*1e+6

        # Convert frequency from GHz to Hz
        STARTFREQ = startFreq * 1e+9
        STOPFREQ = stopFreq * 1e+9

        # alpha is polarization
        alpha = 4 * math.pi * (RADIUS ** 3) * (self.ep - self.em) / (self.ep + 2 * self.em)

        #wc is omiga of the mode(Hz)
        wc = 2*math.pi*c/wavelength

        # A mode split
        g = -alpha*self.em*(FR**2)*wc/(2*self.ec*self.vc)
        
        # A differential damping rate of the two modes
        GamaR = (self.em**(5/2))*(np.abs(alpha)**2)*(FR**2)*(wc**4)/(6*math.pi*(c**3)*self.ec*self.vc)

        #freq is x-Axis, spectrumBefore and spectrumAfter are y-Axis
        freq = np.linspace(STARTFREQ, STOPFREQ, number, dtype="float32")
        spectrumBefore = (freq**2 + 0.25*((Kapa0-Kapa1)**2))/(freq**2 + 0.25*((Kapa0+Kapa1)**2))

        beta = freq*complex(0,-1) + g*complex(0,1) + (Kapa0+Kapa1+GamaR)/2
        spectrumAfter  = np.abs(1-Kapa1*beta/(beta**2-((g*complex(0,1)+GamaR/2))**2))**2

        return [freq, spectrumBefore, spectrumAfter]

    # Perform normalization for the spectrum to prevent neural from saturation
    def normalize(self, v, scale = 0.00001):
        norm =scale * (v-np.mean(v))/np.square(np.var(v))
        return norm

    # Find whether the mode split happen
    def findValley(self, input):
        number = 0
        max = np.max(input)
        min = np.min(input)
        judgeLevel = max - (max - min)/5
        for idx in range(1, len(input) - 1):
             if input[idx - 1] > input[idx] < input[idx + 1] and input[idx] < judgeLevel:
                number = number+1

        return number


if __name__ == '__main__':
    resonance = SingleParticle()
    resonance.__init__()
    freq, spectrumBefore, spectrumAfter = resonance.calculateSpectrum(
        startFreq = -5, stopFreq = 5, number =1000, R = 150.0/scale_R, fr = 200.0/scale_FR,
        k0 = 20.0/scale_K0, k1 = 20.0/scale_K1, wavelength = 1550e-9)

    number = resonance.findValley(spectrumAfter)

    print("valley number: %d" % number)

    plt.plot(freq,spectrumBefore,'-b',label='Spectrum Before')
    plt.plot(freq,spectrumAfter, '-r',label='Spectrum After')
    plt.xlabel('Frequency(GHz)')
    plt.ylabel('Normalized intensity')
    plt.legend()
    plt.show()

