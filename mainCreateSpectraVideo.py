import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import tracking
import videoSource
import frameDecoration
import dataLogger
import spectraExtract

def identityTransform(coords):
	return coords

video = videoSource.VideoSource("MF_AlZr.avi", skip=280)
tracker = tracking.MultiObjectTracker()
decorator = frameDecoration.FrameDecorator()
logger = dataLogger.Logger()

matplotlib.use('TkAgg')
#graph = plt.figure()
fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(9, 11.5))
fig.canvas.manager.window.wm_geometry("+%d+%d" % (0, 0))

outVideo = None


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

	#fig.clf()

	graphNum = 0
	for pId, code, s in spectra:
		if graphNum > 11:
			break
		#print(str(pId) + ": " + str(code))
		color = "green"
		if code == 1:
			color = "blue"
		if code == 2:
			color = "red"
		axs[int(graphNum / 3), graphNum % 3].clear()
		axs[int(graphNum / 3), graphNum % 3].plot(s, label=("" + str(pId)), color=color)
		axs[int(graphNum / 3), graphNum % 3].set_title("ID: " + str(pId))
		graphNum += 1
	while graphNum < 12:
		axs[int(graphNum / 3), graphNum % 3].clear()
		axs[int(graphNum / 3), graphNum % 3].set_title("")
		graphNum += 1
		print()
	#logger.logData(tracker, frameNum)
	fig.suptitle("Frame Number: " + str(frameNum))
	plt.draw()
	fig.tight_layout()
	fig.canvas.flush_events()


	if outVideo is None:
		s = fig.get_size_inches()*fig.dpi
		outVideo = cv2.VideoWriter("spectra1.mp4", cv2.VideoWriter_fourcc(*'DIVX'), 10, (int(s[0]), int(s[1])))
	img = np.array(fig.canvas.buffer_rgba())
	img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
	outVideo.write(img)

	plt.pause(0.01)


video.destroy()
cv2.destroyAllWindows()
outVideo.release()