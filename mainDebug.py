import cv2

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

while True:
	frame, frameNum = video.getFrame()
	if frame is None:
		print("Video Done")
		break

	tracker.processImage(frame)
	spectra = spectraExtract.extractRawSpectra(frame, tracker.getTrackingData())
	cv2.imshow('frame', decorator.decorateFrame(frame, tracker, frameNum, identityTransform, showDebugInfo=True, showOccluded=True, showPath=False))

	logger.logData(tracker, frameNum)

	if cv2.waitKey(0) & 0xFF == ord('q'):
		break

video.destroy()
cv2.destroyAllWindows()