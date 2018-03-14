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


from testapp import trainer
CRLF = "" + chr(10) + chr(13)

@app.route('/train', methods=['GET', 'POST'])
def train():
	print("train()")
	imageURL = postValue('imageURL', None)
	numImages = int(postValue('numImages', 9999))
	print("imageURL=" + imageURL)
	print("numImages=" + str(numImages))
	print("Image URL = " + imageURL)
	output = trainer.trainer(imageURL, numImages)
	return output
