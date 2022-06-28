import numpy as np
import cv2

mappingTransform = np.array([[1, 0, -377], [0, 1, -6], [0, 0, 1]])
spectraLineAngle = 2.2 #Degrees, positive because in image plane, y axis is "down"

def mapParticleToSpectra(x, y):
	coords = np.array([x, y, 1])
	mapped = np.matmul(mappingTransform, coords)
	return mapped[0:2]

def extractRawSpectra(frame, trackingData):
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	blured = cv2.GaussianBlur(gray, (11, 11), 0, 0)
	sharpened = cv2.addWeighted(gray, 5, blured, -4, 0)

	cv2.rectangle(sharpened, (896, 0), (896 - 200, 448), 0, -1)  # Block the spectra

	binary = cv2.threshold(sharpened, 10, 255, cv2.THRESH_BINARY)[1]

	numRegions, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8, cv2.CV_32S)

	data = []
	for i in range(numRegions):
		if i == 0:  # Skip background
			continue
		x = stats[i, cv2.CC_STAT_LEFT]
		y = stats[i, cv2.CC_STAT_TOP]
		w = stats[i, cv2.CC_STAT_WIDTH]
		h = stats[i, cv2.CC_STAT_HEIGHT]
		(cX, cY) = centroids[i]

		if w == 1 and h == 1:
			continue
		data.append((x, y, w, h, cX, cY))
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)

	#for particleId, (x, y, w, h, cX, cY), occCount in trackingData:
	#	return

	return frame