import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin
from objectClassifier.utils.helper import decodeImage
from objectClassifier.pipeline.prediction import PredictionPipeline

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        self.filename = 'inputImageDR.jpg'
        self.classifier = PredictionPipeline(self.filename)

@app.route('/',methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route('/train',methods=['GET','POST'])
@cross_origin()
def train():
    os.system('python main.py')
    return "Training Completed"

@app.route('/predict',methods=['POST'])
@cross_origin()
def predict():
    image = request.json['image']
    decodeImage(image, clientApp.filename)
    result = clientApp.classifier.predict()
    return jsonify(result)

if __name__=='__main__':
    clientApp = ClientApp()

    app.run(host='0.0.0.0',port=8080)



