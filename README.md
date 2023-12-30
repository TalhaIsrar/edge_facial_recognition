Face Detection is one of the most common and yet most complex deep
learning problem. 
In this repo we, explore two approaches to solve the problem. 
1. A pipeline of Yolov5 + SSD using FaceNet (in Keras)
2. A pipeline of Yolov5 +  Eigenfaces

Yolo is used for face detection in the webcam feed and faces are extracted. 
Then the faces are passed to the second stage SSD/Eigenfaces which is used 
to identify the face from the existing databases

All files including weights can be downloaded from:
https://drive.google.com/drive/u/0/folders/1-0rY0h2wT3T2dWBA3SY3-h26AT217BH5

Yolov5 Base Code
This model is built upon a Yolov5 model that can be trained for a custom
network. This model was paired wtih pre trained weight for Face
detection in an image which was further optimized for the faces we
required to work on. The original (https://github.com/ultralytics/yolov5) can be used if any other
network is to be built.

Install
Copy the required code in a directory. The requirements.txt file
contains all the required packages in a
[**Python\>=3.7.0**](https://www.python.org/) environment, including
[**PyTorch\>=1.7**](https://pytorch.org/get-started/locally/).

First run the following file
requirements.txt file

Now to run normal face detection use the following command:
python detect.py --weights weights/yolov7-lite.pt --source 0

For facial identification using ssd run
python detect_ssd.py --weights weights/yolov7-lite.pt --source 0

For facial identification using eigenfaces run
python detect_eigen.py --weights weights/yolov7-lite.pt --source 0

To train the eigenfaces use the following command
python train_eigenface.py

We were able to achieve around 30 fps on i5 12th Gen CPU with 8GB Ram using eigenfaces

Using a raspberry pi 4b, 8 GB Ram we were able to achieve 3 fps.


