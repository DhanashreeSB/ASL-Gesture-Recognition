import cv2
import numpy as np
from  pynput import mouse, keyboard
from pynput.keyboard import Key
import os
import keras
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time
import base64
import os
import io
from keras.models import Model, load_model
from PIL import Image
from flask import request
from flask import jsonify
from flask import Flask
from flask_cors import CORS, cross_origin
import imutils
import switch
from keras.preprocessing.image import img_to_array


model = None
img_rows,img_cols=125, 57
frames = []
num=[8]
max1 =1
real_index = 7
instruction = 'no Gestrue'
pre =0
X_tr=[]
num_classes = 8
text=""
count=0
class_file = 'class_gest.csv'
with open(class_file) as f:
    classes = f.readlines()
classes = [c.strip() for c in classes]

app = Flask(__name__)
CORS(app)

def get_model():
	global model
	model1_path = "/home/dhanashree/MyFolder/trainedModels/vid/acc4/3DCNN+3LSTM_vid_3"
	model = load_model(model1_path)
	print(" * Model loaded!")


print(" * Loading Keras model...")
get_model()

@app.route("/realtime", methods=["POST"])
def realtime():
	global frames
	global img_rows
	global img_cols
	global num
	global max1
	global real_index
	global instruction
	global pre
	global X_tr
	global num_classes
	global text
	global count
	print(count)
	count=count+1
	message = request.get_json(force=True)
	encoded = message['image']
	decoded = base64.b64decode(encoded)
	image = Image.open(io.BytesIO(decoded))
	image = img_to_array(image)
	image=cv2.resize(image,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	frames.append(image)
	input1=np.array(frames)
	text=""
	X_tr=[]
	if input1.shape[0]==36:
		frames = []
		print("shape=",input1.shape)
		X_tr.append(input1)
		X_train= np.array(X_tr)
		# print(X_train.shape)
		train_set = np.zeros((1, 36, img_cols,img_rows,3))
		train_set[0][0:35][:][:][:]=X_train[0,0:35,:,:,:]
		train_set = train_set.astype('float32')
		train_set -= 118.93692
		train_set /= 136.06308
		result_1 = model.predict(train_set)
		#print(result)
		num = np.argmax(result_1,axis =1)
		max1 = np.max(result_1, axis = 1)
		print("num[0]=",num[0])
		print("class=",classes[int(num[0])])
		input1=[]
		real_index = switch.index_threshhold(max1, int(num[0]),pre)
		pre = int(num[0])
		text = classes[int(num[0])]
		count=0
	return text    
