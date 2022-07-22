import cv2
import matplotlib
import matplotlib.pyplot as plt

import tracking
import videoSource
import frameDecoration
import spectraExtract

def identityTransform(coords):
	return coords

video = videoSource.VideoSource("MF_AlZr.avi", skip=250, end=-1)
tracker = tracking.MultiObjectTracker()
decorator = frameDecoration.FrameDecorator()

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

	tracker.processImage(frame, frameNum)
	f, spectra = spectraExtract.extractRawSpectra(frame)
	cv2.imshow('frame', decorator.decorateFrame(frame, frameNum, identityTransform, showDebugInfo=True, showOccluded=True, showPath=False))
	cv2.imshow('Rotated', cv2.resize(f[:300, :700], (1400, 600)))

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