import cv2
import tracking
import colorId

def decorateFrame(f, track, num):
	cv2.putText(f, "Frame Num: " + str(num), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA, False)
	cv2.putText(f, "Last ID used: " + str(track.getPreviouslyUsedId()), (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA, False)

	boundingData = track.getTrackingData()
	for particleId, (x, y, w, h, cX, cY) in boundingData:
		cv2.rectangle(f, (x, y), (x + w, y + h), colors.getColorOfId(particleId), 1)
		cv2.putText(f, str(particleId), (x - 15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA, False)

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

while True:
	ret, frame = video.read()

	if not ret:
		print("Bad read")
		break

	#cv2.imshow('source', frame)


	detectedFrame = frame.copy()
	showDetectedFrame(detectedFrame, tracking.detectObjects(frame))
	cv2.imshow('detected', detectedFrame)

	tracker.processImage(frame)
	cv2.imshow('frame', decorateFrame(frame, tracker, frameNum))

	if cv2.waitKey(0) & 0xFF == ord('q'):
		break
	frameNum += 1

# When everything done, release the capture
video.release()
cv2.destroyAllWindows()