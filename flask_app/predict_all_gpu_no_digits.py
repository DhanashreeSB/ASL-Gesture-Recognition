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
    for layer in model.layers[:-24]:
        layer.trainable = False
    model.add(Dense(26, activation='softmax'))
    set_session(sess)
    model.load_weights('A-to-Y-space-del_cloud1.h5')
    #model=load_model('A-to-Y-asldata.h5')
    model._make_predict_function()
    graph = tf.get_default_graph()
    print(" * Model loaded!")


def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image        


print(" * Loading Keras model...")
get_model()

@app.route("/foregpu", methods=["POST"])
def fore():
    global image_var
    global temp
    global previouslabel
    global text
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
