import face_recognition
import pickle
import numpy as np
from PIL import Image, ImageDraw
import uuid
import cv2

def face_detector():
	with open('C:\\Users\\Rohan\\Desktop\\DhwaniRIS\\Ashoka Face match\\dataset_faces.dat', 'rb') as f:
		all_face_encodings = pickle.load(f)

	# Grab the list of names and the list of encodings
	face_names = list(all_face_encodings.keys())
	face_encodings = np.array(list(all_face_encodings.values()))


	# Get a reference to webcam 
	video_capture = cv2.VideoCapture(1)
	print(video_capture)

	# Initialize variables
	face_locations = []

	while True:
	    # Grab a single frame of video
	    ret, frame = video_capture.read()
	    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
	    rgb_frame = frame[:, :, ::-1]
	    # Find all the faces in the current frame of video
	    unknown_face = face_recognition.face_encodings(rgb_frame)
	    face_locations = face_recognition.face_locations(rgb_frame)



	    for(top, right, bottom, left), unknown_face in zip(face_locations, unknown_face):
	    	matches = face_recognition.compare_faces(face_encodings, unknown_face, tolerance = 0.5)
	    	value = face_recognition.face_distance(face_encodings, unknown_face)
	    	#print(matches,face_names)

	    	name = "Unknown Person"
	    	minimum = np.amin(value)
	    	#print(minimum)

	    	if minimum <= 0.5:
	    		name = face_names[np.argmin(value)]
	    	else:
	    		name = name

	    	# Draw box
	    	cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
	    	font = cv2.FONT_HERSHEY_DUPLEX
	    	cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

	    cv2.imshow('Video', frame)
	    # Hit 'q' on the keyboard to quit!
	    if cv2.waitKey(1) & 0xFF == ord('q'):
	    	break
	    # Release handle to the webcam
	video_capture.release()
	cv2.destroyAllWindows()
	    

if __name__ == '__main__':

	#img = face_recognition.load_image_file("C:\\Users\\Rohan\\Desktop\\DhwaniRIS\\Ashoka Face match\\images\\6.jpeg")
	#face_verification(img).show()
	face_detector()



   