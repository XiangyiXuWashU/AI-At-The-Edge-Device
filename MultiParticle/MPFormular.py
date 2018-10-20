import numpy as np
import math
import cmath
import matplotlib

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

# light speed
c = 3e8

# Scale parameters for training
scale_R  = 1.0
scale_FR = 1.0
scale_K0 = 1.0
scale_K1 = 1.0
scale_KX = 1.0

# Multi Particle Sensing Class
class MultiParticle:
    def __init__(self, ep=1.59 ** 2, em=1, ec=1.45 ** 2, vc=1.47e-16):
        self.ep = ep
        self.em = em
        self.ec = ec
        self.vc = vc

    def calculateSpectrum(self, R, kx, fr, k0=200.0, k1=40.0, startFreq=-5.0, stopFreq=5.0, number=300,
                          wavelength=1550e-9):

        # scale is for training balance
        Kapa0 = scale_K0 * k0 * 1e+6
        Kapa1 = scale_K1 * k1 * 1e+6

        # Convert frequency from GHz to Hz
        STARTFREQ = startFreq * 1e+9
        STOPFREQ = stopFreq * 1e+9

        # Parameters refer to PRA 83, 023803 (2011)
        gS = 0.0
        GamaS = 0.0
        gn0Sum = 0.0
        Gaman0Sum = 0.0

        for i in range(len(R)):
            RADIUS = scale_R * R[i] * 1e-9

            # Scale fr
            FR = scale_FR * fr[i] * 1e-3

            # Scale kx
            KX = scale_KX * kx[i]

            # alpha is polarization
            alpha = 4 * math.pi * (RADIUS ** 3) * (self.ep - self.em) / (self.ep + 2 * self.em)

            # wc is omiga of the mode(Hz)
            wc = 2 * math.pi * c / wavelength

            # Mode split
            gn0 = -alpha * (FR ** 2) * wc / (2 * self.vc)
            gn0Sum += gn0
            gS += gn0 * cmath.exp(2*complex(0, 1)*KX)

            # Mode broadening
            Gaman0 = (np.abs(alpha) ** 2) * (FR ** 2) * (wc ** 4) / (6 * math.pi * ((c/math.sqrt(self.ec)) ** 3) * self.vc)
            Gaman0Sum +=Gaman0
            GamaS += Gaman0 * cmath.exp(2*complex(0, 1)*KX)

        yita = cmath.sqrt(complex(0, 1)*gS + 0.5*GamaS)/\
               cmath.sqrt(complex(0, 1)*(gS.conjugate()) + 0.5*(GamaS.conjugate()))

        gPlus = gn0Sum + (yita*gS.conjugate()).real + 0.5 * (yita*GamaS.conjugate()).imag
        gMinus = gn0Sum - (yita*gS.conjugate()).real - 0.5 * (yita*GamaS.conjugate()).imag

        GamaPlus = Gaman0Sum + (yita*GamaS.conjugate()).real + 2.0 * (yita*gS.conjugate()).imag
        GamaMinus = Gaman0Sum - (yita*GamaS.conjugate()).real - 2.0 * (yita*gS.conjugate()).imag

        # freq is x-Axis, spectrum is y-Axis
        freq = np.linspace(STARTFREQ, STOPFREQ, number, dtype="float32")
        spectrum = np.abs(1 - 0.5*Kapa1*(1/(complex(0, 1)*(-freq + gPlus)+(Kapa0 + Kapa1 + GamaPlus)/2)
            +1/(complex(0, 1)*(-freq + gMinus)+(Kapa0 + Kapa1 + GamaMinus)/2))) ** 2

        return [freq, spectrum]
    # Perform normalization for the spectrum to prevent neural from saturation
    def normalize(self, v, scale=0.00001):
        norm = scale * (v - np.mean(v)) / np.square(np.var(v))
        return norm

    # Find whether the mode split happen
    def findValley(self, inputX, inputY):
        number = 0
        valley = []
        max = np.max(inputY)
        min = np.min(inputY)
        judgeLevel = max - (max - min) / 5
        for idx in range(1, len(inputY) - 1):
            if inputY[idx - 1] > inputY[idx] < inputY[idx + 1] and inputY[idx] < judgeLevel:
                number = number + 1
                valley.append(inputX[idx])

        return valley


if __name__ == '__main__':
    resonance = MultiParticle()
    resonance.__init__()

    # 1 Particle + 1 Particle
    freq, spectrumBefore = resonance.calculateSpectrum(R = [100/scale_R],
                                kx = [0],
                                fr = [200.0/scale_FR],
                                k0=20.0/scale_K0 , k1=20.0/scale_K1 ,startFreq=-5.0, stopFreq=5.0,
                                number=1000, wavelength=1550e-9)

    freq, spectrumAfter = resonance.calculateSpectrum(R = [100/scale_R, 100/scale_R],
                                kx = [0, 0.5*math.pi/scale_KX],
                                fr = [200.0/scale_FR, 200.0/scale_FR],
                                k0=20.0/scale_K0 , k1=20.0/scale_K1 ,startFreq=-5.0, stopFreq=5.0,
                                number=1000, wavelength=1550e-9)

    # # 2 Particle + 1 Particle
    # freq, spectrumBefore = resonance.calculateSpectrum(R = [100/scale_R, 100/scale_R],
    #                             kx = [0,0.3*math.pi/scale_KX],
    #                             fr = [200.0/scale_FR, 200.0/scale_FR],
    #                             k0=20.0/scale_K0 , k1=20.0/scale_K1 ,startFreq=-5.0, stopFreq=5.0,
    #                             number=1000, wavelength=1550e-9)
    #
    # freq, spectrumAfter = resonance.calculateSpectrum(R = [100/scale_R, 100/scale_R, 100/scale_R],
    #                             kx = [0,0.3*math.pi/scale_KX, 0.1*math.pi/scale_KX],
    #                             fr = [200.0/scale_FR, 200.0/scale_FR, 200.0/scale_FR],
    #                             k0=20.0/scale_K0 , k1=20.0/scale_K1 ,startFreq=-5.0, stopFreq=5.0,
    #                             number=1000, wavelength=1550e-9)

    numberBefore = len(resonance.findValley(freq, spectrumBefore))
    numberAfter = len(resonance.findValley(freq, spectrumAfter))

    print("valley number before: %d" % numberBefore)
    print("valley number after: %d" % numberAfter)

    print("valley0 before: %f" % resonance.findValley(freq, spectrumBefore)[0])
    print("valley1 before: %f" % resonance.findValley(freq, spectrumBefore)[1])

    plt.plot(freq, spectrumBefore, '-b', label='Spectrum Before')
    plt.plot(freq, spectrumAfter, '-r', label='Spectrum After')
    plt.xlabel('Frequency(GHz)')
    plt.ylabel('Normalized intensity')
    plt.legend()
    plt.show()
