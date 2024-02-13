import cv2
from deepface import DeepFace
import matplotlib.pyplot as plt
cap = cv2.VideoCapture(0)

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while(True):
	
	
	ret, frame = cap.read()
	predictions = DeepFace.analyze(frame)
	if(predictions['gender']=='Man'):
		predictions['gender'] = 'male'
	

	
	font = cv2.FONT_HERSHEY_SIMPLEX
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	cv2.putText(frame, "Emotion: " + predictions['dominant_emotion'],(10,50), font, 2, (51,0,255), 2, cv2.LINE_4);
	cv2.putText(frame, "Gender: " + predictions['gender'],(20,100), font, 1, (51,0,255), 2, cv2.LINE_4);
	cv2.imshow('frame', gray)

	
	faces = faceCascade.detectMultiScale(
		gray,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(30, 30)
	)

	print("Found {0} faces!".format(len(faces)))
	enforce_detection = False

	for (x, y, w, h) in faces:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

	cv2.imshow('frame', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
