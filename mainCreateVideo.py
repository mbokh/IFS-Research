import cv2
import numpy as np

import videoSource
import tracking
import frameDecoration

def coordTransform(coords):
	return (coords[0] - 496) * 4, (coords[1] + 10) * 4, coords[2] * 4, coords[3] * 4, (coords[4] - 496) * 4, (coords[5] + 10) * 4

video = videoSource.VideoSource("MF_AlZr.avi", skip=250)
tracker = tracking.MultiObjectTracker()
decorator = frameDecoration.FrameDecorator()

outVideo = cv2.VideoWriter("trackingVidWithoutOcclusion16x9.mp4", cv2.VideoWriter_fourcc(*'DIVX'), 20, (1600, 900))

while True:
	frame, frameNum = video.getFrame()
	if frame is None:
		print("Video Done")
		break

	tracker.processImage(frame)

	newFrame = np.zeros((225, 400, 3), np.uint8)
	newFrame[10:, :] = frame[:215, -400:]
	resized = cv2.resize(newFrame, (1600, 900))

	outVideo.write(decorator.decorateFrame(resized, tracker, frameNum, coordTransform, showOccluded=False, showPath=True))

	if frameNum % 20 == 0:
		print(frameNum)

video.destroy()
outVideo.release()