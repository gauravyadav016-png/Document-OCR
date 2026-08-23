from datetime import datetime
from flask import Flask, jsonify, request, render_template_string
import cv2
from PIL import Image
import pytesseract
import re
import spacy
import numpy as np

app = Flask(__name__)

class OCR:
    def __init__(self,config = None):
        if config is None:
            config = {'orient' : True,'skew' : True,'crop': True,'contrast' : True,'psm': [3,4,6],'mask_color': (0, 165, 255), 'brut_psm': [6]}
        self.config = config
        self.nlp = spacy.load('model')
        self.nlp2 = spacy.load('modelback')
        self._height = 1000
        self._rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        self._sq_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
        
    def smooth(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (3, 3), 0)

    def find_dark_regions(self, image):
        return cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, self._rect_kernel)

    def preprocess_image(self, image):
        # Convert PIL image to OpenCV format
        image_cv = np.array(image)
        # Preprocess the image
        smoothed = self.smooth(image_cv)
        dark = self.find_dark_regions(smoothed)
        # thresh = self.apply_threshold(smoothed)
        # Convert preprocessed image back to PIL format
        preprocessed_image = Image.fromarray(dark)
        return preprocessed_image
    
    def get_text(self, image):
        # preprocessed_image = self.preprocess_image(image)
        text = pytesseract.image_to_string(image, config = None)
        print(text)
        return text
    
    def get_dob(self,image):
        preprocessed_image = self.preprocess_image(image)
        text = pytesseract.image_to_string(preprocessed_image, config = None)
        return text
    
    def get_textback(self, image):
        preprocessed_image = self.preprocess_image(image)
        text = pytesseract.image_to_string(preprocessed_image, config = None)
        # print(f'OCR Output: {text}')
        return text
        # Extracting address from Aadhar card
        # match = re.search(r'(Address:.*?\b\d{6}\b)', text, re.DOTALL)
        # if match:
        #     address = match.group(1).strip()
        #     return address
        # else:
        #     return None
        
    def get_aadhaar_number(self,image):
        # Custom configuration for extracting Aadhaar number
        custom_config = r'--oem 3 --psm 6'
        # preprocessed_image = self.preprocess_image(image)
        text = pytesseract.image_to_string(image, config=custom_config)
        # print(text)
        
        cleaned_text = re.sub(r'\s', '', text)
        # Using regex to find a 12-digit number in the extracted text
        aadhaar_number_matches = re.findall(r'\d{12}', cleaned_text)
        # print(aadhaar_number_matches)
        if aadhaar_number_matches:
            return aadhaar_number_matches[0]  # Return the first 12-digit number found
        else:
            return None  # Return None if no 12-digit number is found
        
    def save_preprocessed_image(self, image, filename):
        # Assuming image is a PIL Image object
        image.save(filename)
        
    def extract_entities(self, text):
        doc = self.nlp(text)
        entities = {ent.label_: ent.text for ent in doc.ents}
        return entities
    
    def extract_address(self, text):
        if text is None:
            return 'No text to process'
        doc = self.nlp2(text)
        address_components = []
        # Define the sequence of components based on your training data
        sequence = ['STREET', 'SCHOOL', 'COLONY', 'CITY', 'STATE', 'PINCODE'] 
        for label in sequence:
            component = next((ent.text for ent in doc.ents if ent.label_ == label), None)
            if component:
                address_components.append(component)
        return ', '.join(address_components) if address_components else 'Address not found'


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        front_file = request.files['front']
        back_file = request.files['back']

        if front_file and back_file:
            front_image = Image.open(front_file)
            back_image = Image.open(back_file)
            form_name = request.form['name']
            
        
            ocr = OCR()

            # Specify the file paths as strings
            front_image_filename = 'static/front_preprocessed.png'
            back_image_filename = 'static/back_preprocessed.png'

            # Call save_preprocessed_image with the images and file paths
            ocr.save_preprocessed_image(front_image, front_image_filename)
            ocr.save_preprocessed_image(back_image, back_image_filename)

            # Get the extracted text
            front_text = ocr.get_text(front_image)
            front_text_dob = ocr.get_text(front_image)
            back_text = ocr.get_textback(back_image)
            aadhaar_number = ocr.get_aadhaar_number(front_image)

            # Extract entities using the trained model
            front_entities = ocr.extract_entities(front_text)
            front_entities_dob = ocr.extract_entities(front_text_dob)
            # print(front_entities)

            # Parsing the front image text
            name_match = re.search(r'(?i)([a-z]+)\s+DOB', front_text)
            dob_match = re.search(r'(0?\d{1,2}/0?\d{1,2}/\d{4})', front_text_dob)

            # dob_match = re.search(r'DOB\s*[:;]\s*(0?\d{1,2}/0?\d{1,2}/\d{4})', front_text_dob)

            gender_match = re.search(r'(?i)(male|female)', front_text)
            
            ner_output=front_entities_dob.get("DOB", "Not Found")
            # dobner = ner_output.split(" ")[2]
            
            name = name_match.group(1) if name_match else "Not found"
            dob = dob_match.group(1) if dob_match else ner_output
            gender = gender_match.group(1).capitalize() if gender_match else front_entities.get("GENDER", "Not found")
            # address = back_text.replace('\n', ' ') if back_text else "Not found"
            aadhaar = aadhaar_number if aadhaar_number else front_entities.get("AADHAAR", "Not Found")
            
            
            address = ocr.extract_address(back_text)
            
            
            accuracy = 0
            # Use the extracted entities to construct the result
            result = {
                "Name": front_entities.get("PERSON", name),
                # "DOB": front_entities_dob.get("DOB", dob),
                "Gender": gender,
                "Aadhaar Number":aadhaar ,
                "DOB":dob,
                "Address": address
            }
            if form_name.lower() == result["Name"].lower(): accuracy += 100
            # Include accuracy in the result
            result["Accuracy"] = f"{accuracy}"
            # Checking the accuracy and updating the result object accordingly
            if accuracy == 0:
                result["Status"] = "Failed"
                result["Message"] = "Data Mismatch"
            else:
                result["Status"] = "Success"
                result["Message"] = "Data Match"

            return jsonify(result)

    html = '''
<!doctype html>
<html lang="en">
  <head>
    <!-- Required meta tags -->
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">

    <!-- Bootstrap CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css" rel="stylesheet">

    <title>Aadhaar OCR App</title>
  </head>
  <body>

    <div class="container">
      <h1 class="mt-5">Upload Aadhaar Card Images</h1>
      <form method="post" enctype="multipart/form-data" class="mt-4">
        <div class="mb-3">
          <label for="name" class="form-label">Name</label>
          <input type="text" class="form-control" name="name" required>
        </div>
        <div class="mb-3">
          <label for="front" class="form-label">Front Image</label>
          <input type="file" class="form-control" name="front" required>
        </div>
        <div class="mb-3">
          <label for="back" class="form-label">Back Image</label>
          <input type="file" class="form-control" name="back" required>
        </div>
        <button type="submit" class="btn btn-primary">Submit</button>
      </form>
    </div>

    <!-- Option 1: Bootstrap Bundle with Popper -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/js/bootstrap.bundle.min.js"></script>
  </body>
</html>
    '''

    return html

if __name__ == '__main__':
    app.run(debug=True,port=5003)
