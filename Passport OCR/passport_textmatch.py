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
 
 <h3>Enter Details</h3>
 <label for="fname">First Name:</label><br>
 <input type="text" id="fname" name="fname"><br>
 
  <label for="lname">Last Name:</label><br>
 <input type="text" id="lname" name="lname"><br>
 
 <label for="dob">Date of Birth:</label><br>
 <input type="date" id="dob" name="dob"><br>
 
 <label for="validt">Valid through:</label><br>
 <input type="date" id="validt" name="validt"><br>
 
 <label for="passportNumber">Passport Number:</label><br>
 <input type="text" id="passportNumber" name="passportNumber"><br>
 
 <label for="nationality">Nationality:</label><br>
 <input type="text" id="nationality" name="nationality"><br>
 
 <input type="submit" value="Submit">
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
        detector = MrzDetector()
        reader = MrzReader()
                
        image2 = detector.read(file_stream)
        cropped = detector.crop_area(image2)
        result = reader.process(cropped)
        print(json.dumps(result))
        
        data = {
            "name": request.form.get("fname"),
            "surname": request.form.get("lname"),
            "expiry_date": request.form.get("validt"),
            "document_number": request.form.get("passportNumber"),
            "nationality": request.form.get("nationality"),
            "birth_date":request.form.get("dob")
        }

        # Fix date format
        expiry_date = datetime.strptime(request.form.get("validt"), '%Y-%m-%d')
        expiry_date = expiry_date.strftime('%y%m%d')
        data["expiry_date"] = expiry_date
        
        dateofbirth = datetime.strptime(request.form.get("dob"), '%Y-%m-%d')
        dateofbirth = dateofbirth.strftime('%y%m%d')
        data["birth_date"] = dateofbirth
        
        accuracy = 0
        for key in data:
            if key in result:
                if data[key].lower() == result[key].lower():
                    accuracy += 1
            else:
                continue
        accuracy = (accuracy / len(data)) * 100
        return jsonify({"Status":"Success","mrz_info":result,"Accuracy":accuracy})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5004, debug=True)