**FACE DETECTION & TRACKING **

**ABSTRACT:**

Face detection is a computer vision technology that helps to
locate/visualize human faces in digital images. This is done by
analyzing the visual input to determine whether a person’s facial
features are present. By analyzing video frames in real-time, the system
identifies faces within the frames and tracks them as they move across
the video stream. The methodology involves utilizing Haar cascades model
for face detection, implementing tracking algorithms like the mean-shift
tracking for continuous monitoring, and visualizing the results through
bounding boxes overlaid on the original frames. The system's performance
is evaluated based on accuracy, speed, and robustness, enabling its
application in surveillance, biometrics, and human-computer interaction.

**Libraries used :** cv2

**OBJECTIVE:**

The objective of the face detection and tracking is to develop a system
that can accurately detect and track human faces in real-time video
streams or static images.

**CONTENTS :**

**Face Detection :**

Face detection is a computer vision technique that involves the
identification and localization of human faces within images or video
streams. It is a crucial first step in numerous applications, including
facial recognition, emotion analysis, biometrics, and augmented reality.

The process of face detection typically involves the use of algorithms
and models that analyze the visual characteristics of an image to
determine the presence and location of faces. These algorithms search
for specific patterns, features, and structures that are commonly
associated with human faces, such as the arrangement of eyes, nose, and
mouth.

**Haar Cascade Frontal Face Algorithm:**

The Haar Cascade Frontal Face algorithm is a popular technique for face
detection in computer vision. It is based on the Haar-like features
approach, which uses a series of rectangular patterns to identify and
localize human faces in images or video streams.

The algorithm starts by training a cascade classifier using a large
dataset of positive and negative samples. Positive samples contain
annotated examples of faces, while negative samples consist of random
images that do not contain faces. The training process, known as
AdaBoost, selects a subset of Haar-like features that are most effective
at distinguishing faces from non-facial regions.

![](media/image1.png){width="2.9583333333333335in"
height="2.9895833333333335in"}

**Detect Multiscale:**

**Syntax:**

faces = face\_cascade.detectMultiScale(src, scalefactor,minNeighbors)

**Explanation:**

**scaleFactor** — It specifies how much the image size is reduced at
each image scale.

**minNeighbors** — It specifies how many neighbors each candidate
rectangle should have to retain it.

**WORKFLOW :**

**CODE IMPLEMENTATION:**

**1) Face Detect**

import cv2

\# Load the pre-trained cascade classifier for face detection

face\_cascade =
cv2.CascadeClassifier('haarcascade\_frontalface\_default.xml')

img = cv2.imread('image.png')

gray\_img = cv2.cvtColor(img, cv2.COLOR\_BGR2GRAY)

\# Perform face detection

faces = face\_cascade.detectMultiScale(gray\_img, scaleFactor=1.1,
minNeighbors=5, minSize=(30, 30))

\# Draw rectangles around the detected faces

for (x, y, w, h) in faces:

cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow('Gray Image',gray\_img)

cv2.imshow('Face Detection',img)

cv2.waitKey(0)

cv2.destroyAllWindows()

**Gray Image**

![](media/image2.png){width="5.90625in" height="2.9800688976377954in"}

**Face Detection**

![](media/image3.png){width="6.322916666666667in" height="3.59375in"}

**2) Face Detection in Video **

import cv2

\# Load the pre-trained face cascade classifier

face\_cascade =
cv2.CascadeClassifier('haarcascade\_frontalface\_default.xml')

\# Open the video file

video = cv2.VideoCapture('video.mp4')

\# Create a loop to process each frame of the video

while True:

ret, frame = video.read()

if not ret:

break

\# Convert the frame to grayscale

gray\_frame = cv2.cvtColor(frame, cv2.COLOR\_BGR2GRAY)

\# Detect faces in the grayscale frame

faces = face\_cascade.detectMultiScale(gray\_frame, 1.3, 4)

\# Draw rectangles around the detected faces

for (x, y, w, h) in faces:

cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

cv2.imshow('Face Detection', frame)

if cv2.waitKey(1) & 0xFF == ord('q'):

break

video.release()

cv2.destroyAllWindows()

![](media/image4.png){width="2.7395833333333335in" height="1.8125in"}
![](media/image5.png){width="2.7604166666666665in"
height="1.8055555555555556in"}

![](media/image6.png){width="2.875in" height="1.8645833333333333in"}
![](media/image7.png){width="2.895209973753281in"
height="1.8756944444444446in"}

![](media/image8.jpeg){width="2.8645833333333335in"
height="1.611288276465442in"}
![](media/image9.jpeg){width="2.8958333333333335in"
height="1.5624606299212598in"}

**2)Face Detection using WebCam**

import cv2

\# Load the pre-trained face cascade classifier

face\_cascade =
cv2.CascadeClassifier('haarcascade\_frontalface\_default.xml')

\# Open the webcam

webcam = cv2.VideoCapture(0)

while True:

(\_, img) = webcam.read()

gray\_img = cv2.cvtColor(img, cv2.COLOR\_BGR2GRAY)

\# Detect faces in the grayscale image

faces = face\_cascade.detectMultiScale(gray\_img, 1.3, 4)

\# Draw rectangles around the detected faces

for (x,y,w,h) in faces:

cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

cv2.imshow('FaceDetection', img)

key = cv2.waitKey(10)

if key == 27:

break

webcam.release()

cv2.destroyAllWindows()

**3)Create Face Database for Face recognition of every individual:**

import cv2

import os

\# File paths and directories

datasets = 'dataset'

sub\_data = 'swaru'

path = os.path.join(datasets, sub\_data)

if not os.path.isdir(path):

os.mkdir(path)

width, height = 130, 100

\# Initialize the face cascade classifier

face\_cascade =
cv2.CascadeClassifier('haarcascade\_frontalface\_default.xml')

\# Open the webcam

webcam = cv2.VideoCapture(0)

count = 1

while count &lt; 31:

print(count)

\_, img = webcam.read()

gray\_img = cv2.cvtColor(img, cv2.COLOR\_BGR2GRAY)

\# Detect faces in the grayscale image

faces = face\_cascade.detectMultiScale(gray\_img, 1.3, 4)

\# Draw a rectangle around the detected face

for (x, y, w, h) in faces:

cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

\# Extract the face region from the grayscale image

face = gray\_img\[y:y + h, x:x + w\]

face\_resize = cv2.resize(face, (width, height))

\# Save the resized face image to disk

cv2.imwrite('%s/%s.png' % (path, count), face\_resize)

count += 1

cv2.imshow('OpenCV', img)

key = cv2.waitKey(10)

if key == 27:

break

print("Dataset obtained successfully")

webcam.release()

cv2.destroyAllWindows()

**Folder :** Dataset

**Sub-Folder :** swaru

**CONCLUSION:**

Face detection and tracking are essential techniques in computer vision
for identifying and locating faces in images or video streams. Face
detection algorithms, such as Haar cascades or deep learning-based
models, analyze input data to detect potential face regions. Face
tracking methods enable the continuous localization of faces across
consecutive frames, facilitating tasks like person identification and
tracking.

Effective face detection and tracking algorithms, coupled with robust
tracking mechanisms, provide the foundation for building advanced
systems that can analyze and interpret human faces in real-time
scenarios.
