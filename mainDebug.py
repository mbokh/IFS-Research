import cv2
import matplotlib
import matplotlib.pyplot as plt

import tracking
from Sources import VideoSource
from Sources import FramesSource
import frameDecoration
import spectraExtract

def identityTransform(coords):
	return coords

video = VideoSource.VideoSource(filename="Al3Zr_SM_30k_Run2.avi", skip=0, end=-1, spectraStart=150, spectraEnd=1023)
tracker = tracking.MultiObjectTracker()
decorator = frameDecoration.FrameDecorator()


matplotlib.use('TkAgg')
graph = plt.figure()
graph.canvas.manager.window.wm_geometry("+%d+%d" % (1000, 0))

plt.ion()
plt.show()
while True:
	frame, frameNum = video.getFrame()
	if frame is None:
		print("Video Done")
		break

	tracker.processImage(frame, frameNum, video)
	f, spectra = spectraExtract.extractRawSpectra(frame, video)
	#cv2.imshow('Original', frame)
	cv2.imshow('frame', decorator.decorateFrame(frame, frameNum, identityTransform, showDebugInfo=True, showOccluded=True, showPath=False))
	#cv2.imshow('Rotated', f)

	plt.clf()

	for pId in spectra:
		code = spectra[pId][2]
		s = spectra[pId][1]
		temperature = spectra[pId][0]
		#print(str(pId) + ": " + str(code))
		color = "green"
		if code == 1:
			color = "blue"
		if code == 2:
			color = "yellow"
		if code == 3:
			color = "purple"
		if code == 4:
			color = "black"
		plt.plot(s, label=("id: " + str(pId) + ", T: " + str(temperature)), color=color)
		plt.legend()

	plt.title("Frame Number: " + str(frameNum))
	plt.draw()
	graph.canvas.flush_events()
	plt.pause(0.01)

	if cv2.waitKey(0) & 0xFF == ord('q'):
		break

video.destroy()
cv2.destroyAllWindows()