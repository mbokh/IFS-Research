import cv2

import tracking
import videoSource
import frameDecoration

def identityTransform(coords):
	return coords

video = videoSource.VideoSource("MF_AlZr.avi", skip=250)
tracker = tracking.MultiObjectTracker()
decorator = frameDecoration.FrameDecorator()

while True:
	frame, frameNum = video.getFrame()
	if frame is None:
		print("Video Done")
		break

	tracker.processImage(frame)
	cv2.imshow('frame', decorator.decorateFrame(frame, tracker, frameNum, identityTransform, showOccluded=True, showPath=False))

	if cv2.waitKey(0) & 0xFF == ord('q'):
		break

video.destroy()
cv2.destroyAllWindows()