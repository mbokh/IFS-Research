import cv2
import matplotlib
import matplotlib.pyplot as plt

import tracking
import videoSource
import frameDecoration
import dataLogger
import spectraExtract

def identityTransform(coords):
	return coords

video = videoSource.VideoSource("MF_AlZr.avi", skip=250)
tracker = tracking.MultiObjectTracker()
decorator = frameDecoration.FrameDecorator()
logger = dataLogger.Logger()

matplotlib.use('TkAgg')
graph = plt.figure()
plt.ion()
plt.show()
while True:
	frame, frameNum = video.getFrame()
	if frame is None:
		print("Video Done")
		break

	tracker.processImage(frame)
	f, spectra = spectraExtract.extractRawSpectra(frame, tracker.getTrackingData())
	cv2.imshow('frame', decorator.decorateFrame(frame, tracker, frameNum, identityTransform, showDebugInfo=True, showOccluded=True, showPath=False))
	cv2.imshow('Rotated', f)

	plt.clf()
	for pId, code, s in spectra:
		plt.plot(s)
		plt.pause(0.002)
	plt.draw()
	#logger.logData(tracker, frameNum)


	if cv2.waitKey(0) & 0xFF == ord('q'):
		break

video.destroy()
cv2.destroyAllWindows()