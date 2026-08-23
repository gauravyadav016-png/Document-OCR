# #FaceMatching API
from PIL import Image, ImageFilter, ImageOps,ImageDraw
from flask import Flask, jsonify, request, redirect
import os.path
import base64
import time
from datetime import datetime
import numpy as np
import pytesseract
import cv2
import numpy
import re
import json
from readmrz import MrzDetector, MrzReader

# You can change this to any folder on your system
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    # Check if a valid image file was uploaded
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        # userId=request.form.get('userId');
        if(file.filename == '' ):
            return jsonify({ "Status":"Error","Response":"Upload Valid File"})
        if file and allowed_file(file.filename):
            # print( request)
            # The image file seems valid!.
            file1 = request.files['file']
            
            # path = os.path.join("./uploaded_files", file1.filename)
            print(file1)
            path = "file1.filename.jpg"
            print(path)
            file1.save(path)
         
            return extractData(path)
    # If no valid image file was uploaded, show the file upload form:
    return '''
    <!doctype html>
    <title>Passport Extraction</title>
    <form method="POST" action="/" enctype="multipart/form-data">
      <h3>Upload Passport</h3><input type="file" name="file">
      <input type="submit" value="Upload">
    </form>
    '''    

    
@app.route('/extractPassport', methods=['POST'])
def extractPassport():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            return jsonify({ "Status":"Error","Response":"File Missing"})

        if file and allowed_file(file.filename):
            image = request.files.get("file")
            img =Image.open(image)
            return extractData(img)
        else:
            return jsonify({ "Status":"Error","Response":"Invalid File Formate"})
    return extractPassport()
 
 
def extractData(file_stream):
        image = Image.open(file_stream,mode="r")
        detector = MrzDetector()
        reader = MrzReader()
                
        image2 = detector.read(file_stream)
        cropped = detector.crop_area(image2)
        result = reader.process(cropped)
        print(json.dumps(result))
        return jsonify({"Status":"Success","mrz_info":result})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5004, debug=True)