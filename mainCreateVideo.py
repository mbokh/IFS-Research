import cv2
import numpy as np

import videoSource
import tracking
import frameDecoration

def coordTransform(coords):
	return (coords[0] - 496) * 3, (coords[1] + 50) * 3, coords[2] * 3, coords[3] * 3, (coords[4] - 496) * 3, (coords[5] + 50) * 3

video = videoSource.VideoSource("MF_AlZr.avi", skip=250)
tracker = tracking.MultiObjectTracker()
decorator = frameDecoration.FrameDecorator()

outVideo = cv2.VideoWriter("trackingVidWithoutOcclusion.mp4", cv2.VideoWriter_fourcc(*'DIVX'), 20, (1200, 900))

while True:
	frame, frameNum = video.getFrame()
	if frame is None:
		print("Video Done")
		break

	tracker.processImage(frame)

	newFrame = np.zeros((300, 400, 3), np.uint8)
	newFrame[50:, :] = frame[:250, -400:]
	resized = cv2.resize(newFrame, (1200, 900))

	outVideo.write(decorator.decorateFrame(resized, tracker, frameNum, coordTransform, showOccluded=False, showPath=True))

	if frameNum % 20 == 0:
		print(frameNum)

video.destroy()
outVideo.release()