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
graph.canvas.manager.window.wm_geometry("+%d+%d" % (0, 0))

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
		print(str(pId) + ": " + str(code))
		color = "green"
		if code == 1:
			color = "blue"
		if code == 2:
			color = "red"
		plt.plot(s, label=("" + str(pId)), color=color)

	#logger.logData(tracker, frameNum)
	plt.title("Frame Number: " + str(frameNum))
	plt.draw()
	graph.canvas.flush_events()
	plt.pause(0.01)

	if cv2.waitKey(0) & 0xFF == ord('q'):
		break

video.destroy()
cv2.destroyAllWindows()