# Face Identification
A python 3 program that uses the face recognition library to detect and identify faces in images and videos (both live streaming and regular). 

## Folder Structure
* __Database__ - contains the image of people's faces that we want to recognize in the other images
* __Identified__ - Final images with faces recognized
* __Test__ - images where the program identifies faces
## Installation

Clone the repository and install the libraries mentioned in *requirements.txt* by running the command line commmand:

    pip3 install -r "requirements.txt"

## Running the Program

 1. Add clear images of faces of all the people that the program should identify to the Database folder
 2.  Run the *encodings_to_database.py* program to create the .dat file
 3. Add test image into the test folder and mention the file name in the load_image_file function 
```python
if __name__ == '__main__':
	img = face_recognition.load_image_file("Test/1.jpg")
	face_verification(img).show()
```
 4. The final image with the identified faces will be saved in the Identified folder
