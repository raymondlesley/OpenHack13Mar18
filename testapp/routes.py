from testapp import app

global_data = ""

@app.route('/')
@app.route('/index')
def index():
	return "Hello, World!"
	global_data = "initialised"
	
@app.route('/test')
def test():
	return "Test OK!" + global_data

from testapp import trainer
@app.route('/train')
def train():
	output = trainer.trainer()
	return "<pre>" + output + "</pre>"
