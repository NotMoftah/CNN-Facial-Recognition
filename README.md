# Real-Time-Facial-Recognition-with-DeepLearning
A real-time facial recognition system through webcam streaming and CNN.


# Abstract
This project aims to recognize faces with CNN implemented by Keras. I also implement a real-time module which can real-time capture user's face through webcam steaming called by opencv. OpenCV cropped the face it detects from the original frames and resize the cropped images to 128x128 image, then take them as inputs of deep leanring model. The model then output a hash/embeddings, which is then used to compare the similarity between the captured face and the stored reference faces in the database.


# Dataset
[Georgia Tech face database](http://www.anefian.com/research/face_reco.htm) I picked this dataset because it's fairly small for simple training to test the network structure.


# Environment
I provide my work environment for references.
```sh
$ CPU : i7-3740qm
$ RAM : 8G
```
The whole image processing/training took around 5 minutes.


# Dependencies
To run the code you'll have to install Tensorflow, Keras, OpenCV and Numpy.


# Training
- Data Preprocessing phase
  > 1. Download the dataset
  > 2. Run the mit_data_preprocessing script
  > 3. Modify the save_to path to whatever you like
  
- Training phase
  > 1. Run the train_face_id script
  > 2. Wait untill the training process finishes. it will save the model automatically afterwards.
  
 # Testing
- Adding the referenc phase
  > 1. Add a simple picture of the referenc person on any directory
  > 2. Add the path of the referenc to the test_face_id script
  
- Real-time testing phase
  > 1. Hook up a camera into your workstation 
  > 2. Run test_face_id script


# Final result
![Real-time test](https://github.com/YuP0ra/CNN-Facial-Recognition/blob/master/README/final_result.png)
