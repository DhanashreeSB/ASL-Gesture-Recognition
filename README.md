# Hand gesture recognition

Communication is an essential and significant task for every human being. The normal people who are able to speak and hear can easily and effortlessly communicate with the normal people but the people who are unable to speak and hear are very difficult to communicate with the normal people. Thus, the speech and hearing impaired people communicate well with the other speech and hearing impaired people by making sign gestures as their communication language. This way of communication is called Sign Language (SL). There are more than 70 million people in the world and around 10 million people across India are suffering from speech and hearing disability problems. Sign language recognition (SLR) is a system or process, in which the computer automatically understands the gestures and interprets them into their equivalent human and or machine recognizable or readable text.

## Objectives

1. Objective of this project is to create a complete system to detect, recognize and
interpret the hand gestures through computer vision
2. Therefore to provide a new low-cost, high speed and color image acquisition system.


## Fine tuning of VGG16 using Keras : 

In order to predict the gesture we have trained the VGG16 pretrained network on 3000 images of each gesture from A to Y and gestures for “delete” and “space” in different lighting conditions and on 255 images of each number gesture. Letters J and Z are excluded while training as they are dynamic gestures. 


## Block diagram for static gesture recognition


![image](https://user-images.githubusercontent.com/37834148/181934240-b55cd4b8-67e0-48eb-9f31-0db59143f71a.png)

### Description:
Webapp by default opens in static mode. Starting the web app enables the webcam to capture images in the intervals of 1 second in static mode. Image is resized and sent to the server running at port 5000 for prediction. If the prediction of the gesture label is the same for 5 continuous images then only the label is sent to the user as a prediction. The predicted label text is then converted into the system's voice. All single labels are cached into a temporary word variable continuously. When the user does the gesture for “space” then, the cached word is given to the user as an output and word variable is reinitialized to empty string for next word.


## Block diagram for dynamic gesture recognition:

![image](https://user-images.githubusercontent.com/37834148/181934300-81cd8044-4c68-4b6a-9a56-e449a96db61a.png)

### Description:

If we want the output of a dynamic gesture, then we will have to switch to dynamic mode in the webapp and click on the Start button to start the real time streaming. When 36 continuous frames are captured, they are sent to the web server running at port 8000. These frames are resized and fed to the joint 3DCNN and LSTM model. The label of gesture with highest prediction probability is sent to the user and converted to the system’s voice. Predicted labels are written inside of a textarea on a webapp.


## Publication:

### Web Based Recognition and Translation of American Sign Language with CNN and RNNWeb Based Recognition and Translation of American Sign Language with CNN and RNN
#### International Journal of Online and Biomedical Engineering · Jan 19, 2021

Publication link: https://doi.org/10.3991/ijoe.v17i01.18585
