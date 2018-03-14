# trainer
import sklearn
from dir_helpers import *
import numpy as npy

def dummyProcess(image):
    # return image.histogram()
    d = image.getdata()
    #print("Image data is", len(d), "x", len(d[0]))
    new_d = npy.array(d)
    new_d = npy.reshape(new_d, [16384*3])
    # print("Image data is", len(new_d), "x 1")
    return new_d

def loadImagesForSciKit(input_dir, process):
    all_images = loadImagesRecursive(input_dir, max_per_dir=100)
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

def trainer():
	d, c = loadImagesForSciKit('../gear_images', dummyProcess)

	num_images = len(d)
	num_pixels = len(d[0])
	print("got", num_images, "x", num_pixels)
	values = npy.reshape(d, [num_images*num_pixels])
	print("data range", max(values), "to", min(values))

	X_train, X_test, y_train, y_test = train_test_split(d, c, test_size=0.33, random_state=42)

	from sklearn.ensemble import RandomForestClassifier
	clf = RandomForestClassifier()

	clf.fit(X_train, y_train)

	y_pred = clf.predict(X_test)

	# print(y_test)
	# print(confusion_matrix(y_test, y_pred))
	output = classification_report(y_test, y_pred) + chr(10) + chr(13)
	# print(output)

	accuracy = accuracy_score(y_test, y_pred)*100
	# print("Accuracy: %2.0f%%" % accuracy)
	output = output + "Accuracy = %2.0f%%" % accuracy

	return output
