import cv2
import matplotlib
import matplotlib.pyplot as plt

from sources import VideoSource
from backend import frameDecoration, spectraExtract, tracking, calib


def identityTransform(coords):
	return coords

video = VideoSource.VideoSource(filename="Al3Zr_SM_30k_Run2.avi", skip=0, end=200, spectraStart=150, spectraEnd=1023, flipLR=calib.FLIP_SOURCES_LR)
tracker = tracking.MultiObjectTracker()
decorator = frameDecoration.FrameDecorator()


matplotlib.use('TkAgg')
graph = plt.figure()
graph.canvas.manager.window.wm_geometry("+%d+%d" % (1000, 0))

plt.ion()
plt.show()

#outVideo = cv2.VideoWriter("testing.mp4", cv2.VideoWriter_fourcc(*'DIVX'), 10, (1000, 1000))

while True:
	frame, frameNum = video.getFrame()
	if frame is None:
		print("Video Done")
		break

	tracker.processImage(frame, frameNum, video)
	f, spectra = spectraExtract.extractRawSpectra(frame, video)
	#cv2.imshow('Original', frame)
	img = decorator.decorateFrame(frame, frameNum, identityTransform, showDebugInfo=True, showOccluded=True, showPath=False)
	#resized = cv2.resize(img[100:500, 600:1000], (1000, 1000))
	cv2.imshow('frame', img)
	#outVideo.write(resized)

	#cv2.imshow('Rotated', f)

	plt.clf()
	plt.gca().set_ylim(0, 1 * (10 ** 12))
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
		plt.plot(s, label=("id: " + str(pId) + ", T: " + str(temperature)))
		plt.legend()

	plt.title("Frame Number: " + str(frameNum))
	plt.draw()
	graph.canvas.flush_events()
	plt.pause(0.01)

	if cv2.waitKey(0) & 0xFF == ord('q'):
		break

#outVideo.release()
video.destroy()
cv2.destroyAllWindows()