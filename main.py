import cv2
import numpy as np

import tracking
import colorId

def decorateFrame(f, track, num):
	cv2.putText(f, "Frame Num: " + str(num), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA, False)
	cv2.putText(f, "Last ID used: " + str(track.getPreviouslyUsedId()), (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA, False)

	boundingData = track.getTrackingData()
	for particleId, (x, y, w, h, cX, cY), occCount in boundingData:
		cv2.rectangle(f, (x, y), (x + w, y + h), colors.getColorOfId(particleId), 1)
		cv2.putText(f, str(particleId), (x - 15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255) if occCount == 0 else (0, 0, 255), 1, cv2.LINE_AA, False)
		if occCount > 0:
			cv2.putText(f, str(occCount), (x - 15, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA, False)

	cv2.putText(f, "Particles: " + str(len(boundingData)), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA, False)

	# cv2.circle(img, (int(cX), int(cY)), 3, (0, 0, 255), -1)
	return f


def showDetectedFrame(f, d):
	for x, y, w, h, cX, cY in d:
		cv2.rectangle(f, (x, y), (x + w, y + h), (0, 0, 255), 1)
	return f

video = cv2.VideoCapture("MF_AlZr.avi")
frameNum = 0

tracker = tracking.MultiObjectTracker()
colors = colorId.ColorId()

for i in range(250):
	ret, frame = video.read()

#outVideo = cv2.VideoWriter("sampleTracking.mp4", cv2.VideoWriter_fourcc(*'DIVX'), 20, (896, 280))
black = np.zeros((100, 896, 3), np.uint8)
while True:
	ret, f = video.read()
	frame = np.zeros((548, 896, 3), np.uint8)
	frame[:100, :] = black
	frame[100:, :] = f

	if not ret:
		print("Bad read")
		break

	#cv2.imshow('source', frame)


	detectedFrame = frame.copy()
	showDetectedFrame(detectedFrame, tracking.detectObjects(frame))
	cv2.imshow('detected', detectedFrame)

	tracker.processImage(frame)
	f = decorateFrame(frame, tracker, frameNum)
	#outVideo.write(f[:280, :])
	cv2.imshow('frame', f)

	if cv2.waitKey(0) & 0xFF == ord('q'):
		break
	#print(frameNum)
	frameNum += 1

# When everything done, release the capture
video.release()
cv2.destroyAllWindows()