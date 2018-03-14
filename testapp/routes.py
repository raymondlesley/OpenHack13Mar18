from testapp import app

global_data = ""

def debug(stuff):
	print(stuff);
	None

@app.route('/')
@app.route('/index')
def index():
	return "Hello, World!"
	global_data = "initialised"
	
@app.route('/test')
def test():
	return "Test OK!" + global_data

from flask import request
def postValue(tag, default):
	debug("in postValue()")
	try:
		value = request.form[tag]
		debug("got value: " + value)
	except:
		return default
	if value:
		return value
	else:
		return default


from testapp import trainer, classifier
CRLF = "" + chr(10) + chr(13)

@app.route('/train', methods=['GET', 'POST'])
def train():
	print("train()")
	numImages = int(postValue('numImages', 9999))
	print("numImages=" + str(numImages))
	output = trainer.trainer(numImages)
	return output

@app.route('/classify', methods=['GET', 'POST'])
def classify():
	print("classify()")
	imageURL = postValue('imageURL', None)
	print("imageURL=" + imageURL)
	output = classifier.classifier(imageURL)
	return output
