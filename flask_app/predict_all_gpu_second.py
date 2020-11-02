import base64
import os
import numpy as np
import io
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
import tensorflow as tf
from PIL import Image
from keras import backend as K
from keras.models import Sequential
from keras.models import load_model
from keras.layers.core import Dense, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import decode_predictions
#from keras.backend import clear_session
from flask import request
from flask import jsonify
from flask import Flask
from flask_cors import CORS, cross_origin
#from keras.applications.mobilenet import DepthwiseConv2D, relu6
from keras.utils.generic_utils import CustomObjectScope
from tensorflow.python.keras.backend import set_session
import cv2
import imutils
from skimage import transform
import tensorflow as tf

#import theano.ifelse

model = None
graph = None
sess = None
previouslabel = None
text = None
temp=0
model1 = None
graph1 = None
sess1 = None
previouslabel1 = None
text1 = None
temp1=0
image_var=None
app = Flask(__name__)
CORS(app)


def get_model():
    global model
    global graph
    global sess
    #clear_session()
    #model=load_model('cat-dog-classify-model.h5')
    #with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6}):
    config = tf.compat.v1.ConfigProto()
    sess = tf.compat.v1.Session(config=config)
    vgg16_model = keras.applications.vgg16.VGG16()
    vgg16_model.layers.pop()
    vgg16_model.outputs = [vgg16_model.layers[-6].output]
    vgg16_model.layers[-1].outbound_nodes = []
    model = Sequential()
    for layer in vgg16_model.layers:
        model.add(layer)
    for layer in model.layers[:-23]:
        layer.trainable = False
    model.add(Dense(26, activation='softmax'))
    set_session(sess)
    model.load_weights('/home/dhanashree/MyFolder/Dev/atoz/flask_app/A-to-Y-space-del_cloud1.h5')
    #model=load_model('A-to-Y-asldata.h5')
    model._make_predict_function()
    graph = tf.get_default_graph()
    print(" * Model loaded!")

def get_digits_model():
    global model1
    global graph1
    global sess1
    #clear_session()
    #model=load_model('cat-dog-classify-model.h5')
    #with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6}):
    config = tf.compat.v1.ConfigProto()
    sess1 = tf.compat.v1.Session(config=config)
    vgg16_model = keras.applications.vgg16.VGG16()
    vgg16_model.layers.pop()
    vgg16_model.layers[-1].outbound_nodes = []
    model1 = Sequential()
    for layer in vgg16_model.layers:
        model1.add(layer)
    for layer in model.layers[:-18]:
        layer.trainable = False
    model1.add(Dense(10, activation='softmax'))
    set_session(sess1)
    model1.load_weights('1_10_digit_train_4.h5')
    #model=load_model('A-to-Y-asldata.h5')
    model1._make_predict_function()
    graph1 = tf.get_default_graph()
    print(" * Digits Model loaded!")


def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image        


print(" * Loading Keras model...")
get_model()
get_digits_model()



@app.route("/foregpu", methods=["POST"])
def fore():
    global image_var
    global temp
    global previouslabel
    global text
    global graph
    global sess
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    im = preprocess_image(image, target_size=(224,224))
    with graph.as_default():
        set_session(sess)
        prediction = model.predict(im).tolist()
        # # convert the probabilities to class labels
        # label = decode_predictions(prediction)
        # # retrieve the most likely result, e.g. highest probability
        # label = label[0][0]
        label = model.predict_classes(im)
        l=None
        if(int(label)<=8):
        	l = chr(int(label)+65)
        elif(int(label)<=23):
        	l = chr(int(label)+66)
        elif(int(label)==24):
        	l = 'space'
        else:
        	l = 'del'	
        print(int(label))	
        print(l)	
        if (l!=None):
        	if(temp==0):
        		previouslabel=l
        	if previouslabel == l:
        		previouslabel=l
        		temp+=1
        	else :
        		temp=0
        	# if(flag):
        	# 	if(temp<=50):
        	# 		break
        	if(temp==5):
        		print("inside 30")
        		text= l
        	else:
        		if(temp<=10):
        			text=""
        		else:
        			temp=0
        	print(text)	
        print(temp)
    return text


@app.route("/number", methods=["POST"])
def number_pred():
    global image_var
    global temp1
    global previouslabel1
    global text1
    global graph1
    global sess1
    global model1
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    im = preprocess_image(image, target_size=(224,224))
    with graph.as_default():
        set_session(sess1)
        prediction = model1.predict(im).tolist()
        label = model1.predict_classes(im)
        l=str(label[0])
        print(type(l)," ",l)
        if (label!=None):
            if(temp1==0):
                previouslabel1=l
            if previouslabel1 == l:
                previouslabel1=l
                temp1+=1
            else :
                temp1=0
            # if(flag):
            #   if(temp<=50):
            #       break
            if(temp1==5):
                print("inside 30")
                text1= l
            else:
                if(temp1<=10):
                    text1=""
                else:
                    temp1=0
            print("text1=",text1) 
        print("temp1",temp1)
    return text1