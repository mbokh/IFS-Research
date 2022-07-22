import numpy as np
import calibration.calib as calibration
from matplotlib import pyplot as plt

def cameraResponse(i):
	return np.exp(-0.5 * np.square( (i - (calibration.pixelEnd/2)) / (calibration.pixelEnd / 5) )) * 10**(-10)

def convertPhysicalToPixel(planckData):
	return planckData * cameraResponse(calibration.pixelX)

def convertPixelToPhysical(pixelData):
	return pixelData / cameraResponse(calibration.pixelX)


c1 = 1.19 / (10**16)
c2 = 0.0144

tempLookup = dict()

def clearDict(): #When starting a new cluster, offsets are different, so have to clear cache, faster than keying on temp only
	tempLookup.clear()

def plancksLaw(wavelengths, T):
	return (c1 / np.power(wavelengths, 5)) / (np.exp(c2 / (wavelengths * T)) - 1)

#Returns the appropriate response-shaped curve in pixel space in both x and y
def createCurvePixelSpace(T, pixelShift, maxOffset):
	'''if not T in tempLookup.keys():
		tempLookup[T] = plancksLaw(calibration.pixelToWavelength(), T)
	return np.pad(tempLookup[T], (pixelShift, maxOffset - pixelShift))
	'''

	if (T, pixelShift) in tempLookup.keys(): #Lookup needs to be float, not int, or else small steps of least-squares will round to same int
		return tempLookup[(T, pixelShift)]
	raw = plancksLaw(calibration.pixelToWavelength(), T)

	shiftedI = np.zeros(calibration.pixelEnd + maxOffset)
	shiftedI[pixelShift:calibration.pixelEnd + pixelShift] = convertPhysicalToPixel(raw)
	tempLookup[(T, pixelShift)] = shiftedI
	return shiftedI

def createCurvePhysicalSpace(T):
	return plancksLaw(calibration.pixelToWavelength(), T)

'''
temps = [2350, 2750, 2350, 2520]
offsets = [0, 10, 40, 50]

curves = [createCurvePixelSpace(temps[i], offsets[i], offsets[-1]) for i in range(len(temps))]
combinedIntensities = sum(curves)

combinedX = np.linspace(0, calibration.pixelEnd + offsets[-1] - 1, calibration.pixelEnd + offsets[-1])
fig, axs = plt.subplots(2, 1)
for c in curves:
	axs[0].plot(combinedX, c)
axs[1].plot(combinedX, combinedIntensities, 'g')
fig.tight_layout()
plt.show()
'''
