import cv2
import numpy as np
import base64
from flask import Flask, request, jsonify, render_template
from deepface import DeepFace

app = Flask(__name__, static_folder="", template_folder="")

@app.route('/')
def home():
    return render_template('index.html')  # Serve index.html

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json['image']
    
    # Decode base64 image
    encoded_data = data.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    try:
        result = DeepFace.analyze(image, actions=['emotion'])
        dominant_emotion = result[0]['dominant_emotion']
        return jsonify({'emotion': dominant_emotion})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
