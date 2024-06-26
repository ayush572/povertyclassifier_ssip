from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import joblib
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input, InceptionV3
import numpy as np
import tensorflow as tf
from tensorflow import Graph
import json
import os

# getting the model

model_graph = Graph()
with model_graph.as_default():
    tf_session = tf.compat.v1.Session()
    with tf_session.as_default():
        classifier_model = joblib.load('./models/best_model_new.joblib')

with open('./models/knn_classes.json','r') as f:
    labelInfo = f.read()
labelInfo = json.loads(labelInfo)


img_height, img_width = 256, 256

# Create your views here.
def index(req):
    context={'a':1}
    return render(req, "index.html", context)

def predictImage(req):
    print(req.FILES['filePath'])
    fileObj = req.FILES['filePath']
    fs = FileSystemStorage()
    filepathnm = fs.save(fileObj.name, fileObj)
    filepathnm = fs.url(filepathnm) # to get full path of image
    testimg = '.'+filepathnm
    img = image.load_img(testimg, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Added an extra dimension for batch size
    img_array = preprocess_input(img_array)

    # Load the InceptionV3 model with pre-trained weights
    inception_v3_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(256, 256, 3))

    # Extract features using the InceptionV3 model
    features = inception_v3_model.predict(img_array)

    # Reshape the features to 2D
    features_2D = features.reshape(features.shape[0], -1)

    # Get the top predicted class
    # classifier_model is the already trained knn classifier model
    with model_graph.as_default():
        with tf_session.as_default():
            predictions = classifier_model.predict(features_2D)

    
    predicted_class = predictions[0]

    predicted_label = labelInfo[str(predicted_class)]

    print('filepathnm is', filepathnm)  


    context = {'filepathnm': filepathnm, 'predicted_class': predicted_label}
    # this context is accessible in frontend
    return render(req, "index.html", context)

def viewDataBase(req):
    listofImages = os.listdir('./media/')
    listofImagesPath=['./media/'+i for i in listofImages]
    context = {'listofImagesPath':listofImagesPath}
    return render(req, "viewDB.html", context)

from .models import Addpred

def save_prediction(request):
    if request.method == 'POST':
        # img_path = request.POST.get('filePath')  # Assuming the file input field is named 'filePath'
        # Assuming the file input field is named 'filePath'
        img_path = request.FILES.get('filePath')
        prediction = request.POST.get('predicted_class')  # Assuming the predicted class is sent as a POST parameter
        print('views.py image', img_path)
        print('views.py predicted_class', prediction)
        Addpred.objects.create(img_path=img_path, prediction=prediction)
        # Optionally, you can redirect the user to another page after saving the data
        # return redirect('some-url-name')  # Replace 'some-url-name' with the name of the URL pattern you want to redirect to
    else:
        # Handle GET requests if needed
        print('Not Saved')
        pass
