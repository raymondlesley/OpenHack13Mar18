# trainer
import sklearn
from dir_helpers import *
import numpy as npy
from sklearn.externals import joblib

def dummyProcess(image):
    # return image.histogram()
    d = image.getdata()
    #print("Image data is", len(d), "x", len(d[0]))
    new_d = npy.array(d)
    new_d = npy.reshape(new_d, [16384*3])
    # print("Image data is", len(new_d), "x 1")
    return new_d

def loadImagesForSciKit(input_dir, process, max_per_dir=9999):
    all_images = loadImagesRecursive(input_dir, max_per_dir=200)
    output_data = []
    classes = []

    for image in all_images:
        # print("from:", image.parent.name + '/' + image.name)
        img = loadImage128x128(image)
        # print("loaded:", img)
        image_data = process(img)
        #filename = output_dir + '/' + str(image_number) + '.jpg'
        dirname = image.parent.name  # class
        # print("Image class ", dirname)
        output_data.append(image_data)
        classes.append(dirname)
    return output_data, classes

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

def classifier(imageURL):

	clf = joblib.load("gear_model.pkl")

	#load Image from URL
	import urllib
	from PIL import Image
	from io import BytesIO
	print("Loading image from " + imageURL)
	import urllib.request
	with urllib.request.urlopen(imageURL) as response:
		r = response.read()	# img_file = urllib.urlopen(imageURL)
		im = BytesIO(r)
		resized_image = sizeImage(Image.open(im), 128, 128)

	print(resized_image.size)
	data = dummyProcess(resized_image)
	print(len(data))
	X_test = dummyProcess(resized_image)
	print("reshaping")
	reshaped=X_test.reshape(1, -1)
	print("Predicting...")
	y_pred = clf.predict(reshaped)
	print(type(y_pred))
	output = "Result : "
	for result in y_pred:
		print("Got: " + result)
		output += result + "; "

	return output
