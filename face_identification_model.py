import face_recognition
import pickle
import numpy as np
from PIL import Image, ImageDraw
import uuid


def face_verification(test_image):
	# Load face encodings
	with open('dataset_faces.dat', 'rb') as f:
		all_face_encodings = pickle.load(f)

	# Grab the list of names and the list of encodings
	face_names = list(all_face_encodings.keys())
	face_encodings = np.array(list(all_face_encodings.values()))
	#print(face_names)
	#print(all_face_encodings)

	# Try comparing an unknown image
	#test_image = face_recognition.load_image_file("C:/Users/HP/face_recognition_examples/img/Dhwani/photos/12.jpg")
	unknown_face = face_recognition.face_encodings(test_image)
	face_locations = face_recognition.face_locations(test_image)

	# Convert to PIL format
	pil_image = Image.fromarray(test_image)

	# Create a ImageDraw instance
	draw = ImageDraw.Draw(pil_image)

	for(top, right, bottom, left), unknown_face in zip(face_locations, unknown_face):
	  matches = face_recognition.compare_faces(face_encodings, unknown_face, tolerance = 0.5)
	  value = face_recognition.face_distance(face_encodings, unknown_face)
	  #print(matches,face_names)
	  
	  name = "Unknown"
	  minimum = np.amin(value)
	  #print(minimum)

	  if minimum <= 0.5:
	  	name = face_names[np.argmin(value)]
	  else:
	  	name = name

	  # Draw box
	  draw.rectangle(((left, top), (right, bottom)), outline=(255,255,0))

	  # Draw label
	  text_width, text_height = draw.textsize(name)
	  draw.rectangle(((left,bottom - text_height - 10), (right, bottom)), fill=(255,255,0), outline=(255,255,0))
	  draw.text((left + 6, bottom - text_height - 5), name, fill=(0,0,0))

	del draw

	# Save image
	pil_image.save('Indentified/1.jpg')

	return(pil_image)	 


if __name__ == '__main__':

	img = face_recognition.load_image_file("Test/1.jpg")
	face_verification(img).show()



   