import csv
import cv2
import numpy as np
from scipy.interpolate import interp1d

#-------------------------------------------------------------------------------------------
#Looking at the AL3Zr_SM_TrippleNotch for wavelength and pixel space positions
#Physical: [29, 281]

#633 nm peak: [456, 281]
#532 nm peak: [567, 280]
#450 nm peak: [652, 278]


pixelPositions = [456, 567, 652] #X axis data
wavelengths = [633, 532, 450] #Y axis data
samples = np.interp([500, 600], pixelPositions, wavelengths)
PIXEL_TO_WAVELENGTH_SCALING = (samples[1] - samples[0]) / 100
offsetToPeak = (567 - 29, 280 - 281) #532 nm peak
SPECTRA_ANGLE = -0.6 #math.degrees(math.atan2(278 - 281, 652 - 456)) According to full spectra image, it is 0.5

fullSpectra = cv2.imread("calibrationData/Cal_Thorlabs_LampScan_30umSlit000001.tif")[450:600, :] #Horizontal strip
rotate_matrix = cv2.getRotationMatrix2D(center=(512, 75), angle=SPECTRA_ANGLE, scale=1)
height, width = fullSpectra.shape[:2]
gray = cv2.cvtColor(cv2.warpAffine(src=fullSpectra, M=rotate_matrix, dsize=(width, height)), cv2.COLOR_BGR2GRAY)


subsection = gray[30:60, :] #Take a chunk of image as the background
backgroundValue = int(np.sum(subsection, axis=None) / (30 * 1024)) #Average value
gray = gray - backgroundValue

binarySpectra = cv2.threshold(gray[:, 200:], 2, 255, cv2.THRESH_BINARY)[1]
binaryParticle = cv2.threshold(gray[:, :200], 15, 255, cv2.THRESH_BINARY)[1]
combinedBinary = gray.copy()
combinedBinary[:, 200:] = binarySpectra
combinedBinary[:, :200] = binaryParticle

numRegions, labels, stats, centroids = cv2.connectedComponentsWithStats(combinedBinary, 8, cv2.CV_32S)

assert(numRegions == 3)
data = []
for i in range(1, 3):
	x = int(stats[i, cv2.CC_STAT_LEFT])
	y = int(stats[i, cv2.CC_STAT_TOP])
	w = int(stats[i, cv2.CC_STAT_WIDTH])
	h = int(stats[i, cv2.CC_STAT_HEIGHT])
	(cX, cY) = centroids[i]

	data.append((x, y, w, h, int(cX), int(cY)))

spectraIndex = 0
if data[0][2] * data[0][3] < data[1][2] * data[1][3]:
	spectraIndex = 1

spectraData = data[spectraIndex]
particleData = data[1 - spectraIndex]


OFFSET_TO_CENTROID = (spectraData[4] - particleData[4], spectraData[5] - particleData[5])
EXTEND_LEFT = spectraData[4] - spectraData[0]
EXTEND_RIGHT = spectraData[2] - EXTEND_LEFT
#Technically this is slightly off because of the initial rotation, but since it's so small, we ignore it: ~400 cos(0.6) ~= 400

vector = gray[spectraData[1]:spectraData[1] + spectraData[3] + 1, spectraData[0]:spectraData[0] + spectraData[2]].sum(axis=0)
measuredCalibration = vector / (spectraData[3] * 8) #vector / (spectraData[3] * particleData[3]) Manually for weird shape


def getWavelengthFromPixelPosition(x):
	return int(PIXEL_TO_WAVELENGTH_SCALING * (x - pixelPositions[1]) + wavelengths[1])

MAX_WAVELENGTH = getWavelengthFromPixelPosition(spectraData[0]) / 1000000000
MIN_WAVELENGTH = getWavelengthFromPixelPosition(spectraData[0] + spectraData[2]) / 1000000000


#Now open known spectra file and only look at range [min wavelength, max wavelength]
knownSpectraWavelengths = []
knownSpectraValues = []
#Values are normalized to peak, so max raw spectra value is 1. Spectrum corresponds to blackbody @2796K
#According to known spectra values, value "1" is associated with wavelength = 1013, and blackbody is 0.998
#Spectra value is:
spectraScaleFactor = 6.953 * (10**11)
with open("calibrationData/KnownSpectraExtracted.csv", 'r') as file:
	reader = csv.reader(file)
	discardFirst = True
	for row in reader:
		if discardFirst:
			discardFirst = False
			continue
		if MIN_WAVELENGTH * 1000000000 <= int(row[0]) <= MAX_WAVELENGTH * 1000000000:
			knownSpectraWavelengths.append(int(row[0]))
			knownSpectraValues.append(float(row[1]) * spectraScaleFactor)

linInterp = interp1d(knownSpectraWavelengths, knownSpectraValues)

interpolatedX = np.linspace(knownSpectraWavelengths[0], knownSpectraWavelengths[-1], num=EXTEND_RIGHT + EXTEND_LEFT, endpoint=True)
interpolatedSpectra = linInterp(interpolatedX)

SYSTEM_RESPONSE = measuredCalibration / interpolatedSpectra


PIXEL_END = EXTEND_RIGHT + EXTEND_LEFT #Exclusive
pixelX = np.linspace(0, PIXEL_END - 1, PIXEL_END)
MIN_TEMP = 2000.0
MAX_TEMP = 3000.0
PIXEL_TO_WAVELENGTH = MIN_WAVELENGTH + (MAX_WAVELENGTH - MIN_WAVELENGTH)*(pixelX / PIXEL_END)


#import matplotlib.pyplot as plt
#plt.plot(measuredCalibration, '--')
#plt.plot(interpolatedSpectra, '-')
#plt.plot(SYSTEM_RESPONSE, '--')
#plt.show()
