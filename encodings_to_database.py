import face_recognition
import pickle
import os 
import glob

all_face_encodings = {}

folder = glob.glob("C:\\Users\\Rohan\\Desktop\\DhwaniRIS\\Ashoka Face match\\test\\*.jpg")

for img in folder:
	image = face_recognition.load_image_file(img)
	face_locations = face_recognition.face_locations(image)
	name = (img.split('C:\\Users\\Rohan\\Desktop\\DhwaniRIS\\Ashoka Face match\\test\\')[1])
	name = name.split('.')[0]
	print(name)
	all_face_encodings[name] = face_recognition.face_encodings(image, num_jitters=1)[0]


with open('dataset_faces.dat', 'wb') as f:
    pickle.dump(all_face_encodings, f)

print(all_face_encodings)