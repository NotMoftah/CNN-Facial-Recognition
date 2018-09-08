# Real-Time-Facial-Recognition-with-DeepLearning
A real-time facial recognition system through webcam streaming and CNN.


# Abstract
This project aims to recognize faces with CNN implemented by Keras. I also implement a real-time module which can real-time capture user's face through webcam steaming called by opencv. OpenCV cropped the face it detects from the original frames and resize the cropped images to 128x128 image, then take them as inputs of deep leanring model. The model then output a hash/embeddings, which is then used to compare the similarity between the captured face and the stored reference faces in the database.


# Dataset
[Georgia Tech face database](http://www.anefian.com/research/face_reco.htm) I picked this dataset because it's fairly small for simple training to test the network structure.


# Environment
I provide my work environment for references.


# Hadware
CPU : i7-3740qm  
RAM : 8G  
the whole image processing/training took around 5 minutes.


# Dependencies
to run the code you'll have to install Tensorflow, Keras, OpenCV and Numpy.
