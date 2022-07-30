# Hand gesture recognition

Communication is an essential and significant task for every human being. The normal people who are able to speak and hear can easily and effortlessly communicate with the normal people but the people who are unable to speak and hear are very difficult to communicate with the normal people. Thus, the speech and hearing impaired people communicate well with the other speech and hearing impaired people by making sign gestures as their communication language. This way of communication is called Sign Language (SL). There are more than 70 million people in the world and around 10 million people across India are suffering from speech and hearing disability problems. Sign language recognition (SLR) is a system or process, in which the computer automatically understands the gestures and interprets them into their equivalent human and or machine recognizable or readable text.

## Objectives

1. Objective of this project is to create a complete system to detect, recognize and
interpret the hand gestures through computer vision
2. Therefore to provide a new low-cost, high speed and color image acquisition system.


## Fine tuning of VGG16 using Keras : 

In order to predict the gesture we have trained the VGG16 pretrained network on 3000 images of each gesture from A to Y and gestures for “delete” and “space” in different lighting conditions and on 255 images of each number gesture. Letters J and Z are excluded while training as they are dynamic gestures. 
