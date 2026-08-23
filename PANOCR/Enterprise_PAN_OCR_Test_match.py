from datetime import datetime
import re
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image, ExifTags
import pytesseract
import cv2
import numpy as np
import os
import spacy
from datetime import datetime

app = Flask(__name__)

# This is the path to the upload directory
app.config['UPLOAD_FOLDER'] = 'uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
# These are the extension that we are accepting to be uploaded
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Load your spaCy NER model
nlp = spacy.load('modelPANENTER')

# For a given file, return whether it's an allowed type or not
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']
@app.route('/', methods=['GET'])
def index():
    return '''
    <!doctype html>
    <title>Upload an Image for OCR</title>
    <h1>Upload an Image for OCR and Enter Details</h1>
    <form method=post enctype=multipart/form-data action="/upload">
      <input type="file" name="file"><br><br>
      Name: <input type="text" name="name"><br><br>
      Date: <input type="date" name="date"><br><br>
      PAN Number: <input type="text" name="pan"><br><br>
      <input type="submit" value="Upload">
    </form>
    '''



# Route that will process the file upload
@app.route('/upload', methods=['POST'])
def upload():
    # Check if the request has the file part and other form data
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Perform OCR on the uploaded image file
        ocr_result = perform_ocr(filepath)

        
        
        # Compare OCR results with user input and prepare the response
        response = {}
        response["OCR_TEXT"] = ocr_result
        # Access additional form fields
        user_name = request.form.get('name', '')
        user_date_str = request.form.get('date', '')
        user_pan = request.form.get('pan', '')
        
        # Validate and format date
        # Convert to datetime object
        try:
            user_date = datetime.strptime(user_date_str, '%Y-%m-%d')
        except ValueError:
            return jsonify({'error': 'Invalid date format'}), 400

        # Convert to DD/MM/YYYY format if needed
        formatted_date = user_date.strftime('%d/%m/%Y')
        # Process the file and get OCR results
        ocr_results = perform_ocr(filepath)  # Assuming this function returns a dictionary with OCR results

    # Calculate accuracy for each field
        name_accuracy = calculate_accuracy(ocr_results.get("Enterprise Name", ""), user_name)
        date_accuracy = calculate_accuracy(ocr_results.get("Date of Establishment", ""), formatted_date)
        pan_accuracy = calculate_accuracy(ocr_results.get("Enterprise PAN", ""), user_pan)

    # Include accuracy in the response
        response = {
            "OCR_TEXT": ocr_results,
            "Accuracy": {
                "Name Accuracy": f"{name_accuracy:.2f}%",
                "Date Accuracy": f"{date_accuracy:.2f}%",
                "PAN Accuracy": f"{pan_accuracy:.2f}%"
            }
        }
        # Remove the uploaded file
        os.remove(filepath)
        return jsonify(response)
    else:
        return jsonify({'error': 'Invalid file type'}), 400
    
def calculate_accuracy(ocr_text, user_input):
    # Simple accuracy calculation based on character match
    if not user_input:  # Prevent division by zero
        return 0 if ocr_text else 100  # If user input is empty, return 0 or 100 based on ocr_text

    matches = sum(1 for o, u in zip(ocr_text, user_input) if o == u)
    return (matches / len(user_input)) * 100

def filter_name(name):
    # Remove anything that isn't a letter or a space
    return re.sub(r'[^a-zA-Z\s]', '', name)

def filter_dob(dob_text):
    # Assuming enterprise dates are in the format dd-mm-yyyy or dd/mm/yyyy
    dob_text = re.sub(r'[^\d/-]', '', dob_text)
    match = re.match(r'(\d{2})[-/](\d{2})[-/](\d{4})', dob_text)
    if match:
        return f"{match.group(1)}/{match.group(2)}/{match.group(3)}"
    else:
        return dob_text
    
def filter_pan(pan_text):
    # Find all matches of the PAN format in the text, allowing for digits in place of letters and vice versa
    matches = re.findall(r'[A-Z0-9]{5}[0-9A-Z]{4}[A-Z0-9]', pan_text.upper())

    corrected_matches = []
    for match in matches:
        corrected = ''

        # Correct the first five alphabetic characters
        for i in range(5):
            char = match[i]
            if char == '0':
                char = 'O'
            elif char == '1' or char == 'l':
                char = 'I'
            elif char == '2':
                char = 'Z'
            elif char == '5':
                char = 'S'
            elif char == '8':
                char = 'B'
            corrected += char

        # Correct the four digits
        for i in range(5, 9):
            char = match[i]
            if char == 'O':
                char = '0'
            elif char == 'I' or char == 'l':
                char = '1'
            elif char == 'Z':
                char = '2'
            elif char == 'S':
                char = '5'
            elif char == 'B':
                char = '8'
            corrected += char

        # Correct the last alphabetic character
        char = match[9]
        if char == '0':
            char = 'O'
        elif char == '1' or char == 'l':
            char = 'I'
        elif char == '2':
            char = 'Z'
        elif char == '5':
            char = 'S'
        elif char == '8':
            char = 'B'
        corrected += char

        corrected_matches.append(corrected)

    return corrected_matches[0] if corrected_matches else "Not Recognized"


def clean_fname(fname):
    # Remove newlines and leading/trailing whitespace
    fname = re.sub(r'\s*\n\s*', ' ', fname)  # Replace newline and surrounding whitespace with a single space
    fname = fname.strip()  # Remove any leading/trailing whitespace
    return fname

def perform_ocr(image_path):
    # Load the image
    image = Image.open(image_path)
    # Ensure image is in the correct orientation based on EXIF data
    try:
        exif_data = image._getexif()
        # Check if EXIF data was found
        if exif_data is not None:
            # Extract EXIF data using a dictionary comprehension
            exif = {ExifTags.TAGS[k]: v for k, v in exif_data.items() if k in ExifTags.TAGS and isinstance(v, (int, str))}
            # Now check for orientation in EXIF data and rotate the image accordingly
            if "Orientation" in exif:
                if exif["Orientation"] == 3:
                    image = image.rotate(180, expand=True)
                elif exif["Orientation"] == 6:
                    image = image.rotate(270, expand=True)
                elif exif["Orientation"] == 8:
                    image = image.rotate(90, expand=True)
        else:
            print("No EXIF data found for image:", image_path)

    except AttributeError as error:
        print("Error accessing EXIF data:", error)

    image_cv = np.array(image)
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    bg=cv2.morphologyEx( gray ,cv2.MORPH_DILATE, rect_kernel)
    out_gray=cv2.divide( gray, bg, scale=255)
    new_min, new_max = 0, 255
    contrast_stretched = cv2.normalize(out_gray, None, alpha=new_min, beta=new_max, norm_type=cv2.NORM_MINMAX)
    
    
    # Define a small structuring element for opening
    kernel_size = (1, 2)  # Adjust the size as needed
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    # Apply opening to separate the words
    opened = cv2.morphologyEx(contrast_stretched, cv2.MORPH_OPEN, kernel)

    # Define erosion kernel size and apply erosion
    erosion_size = (1, 2)  # Adjust the size as needed for your specific image
    erosion_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, erosion_size)
    eroded = cv2.erode(opened, erosion_kernel, iterations=1)
    # Perform OCR using pytesseract
    extracted_text = pytesseract.image_to_string(eroded,lang='eng',config = r'--oem 3 --psm 3')
    print(extracted_text)
# Now apply NER model on the extracted text
    entities = extract_entities(extracted_text)
    
    # Initialize variables for entities
    name = entities.get("NAME", None)
    dob = entities.get("DATE", None)
    pan = entities.get("PAN", None)
    print(dob)
    # Create a JSON response using the variables
    responseOCR = {
        "Enterprise Name": name if name is not None else "Not Recognized",
        "Date of Establishment": dob if dob is not None else "Not Recognized",
        "Enterprise PAN": pan if pan is not None else "Not Recognized"
    }
    
    # Assume ocr_result is a dictionary containing the OCR results
    name = filter_name(responseOCR.get("NAME", ""))
    pan = filter_pan(responseOCR.get("PAN", ""))
    filtered_result = {
        "Enterprise Name": filter_name(responseOCR.get("Enterprise Name", "")),
        "Date of Establishment": filter_dob(responseOCR.get("Date of Establishment", "")),
        "Enterprise PAN": filter_pan(responseOCR.get("Enterprise PAN", ""))
    }

    return filtered_result
    
    

def extract_entities(text):
    # Process the text with the NER model
    doc = nlp(text)
    # Initialize all entities with None
    entities = {"NAME": None, "DATE": None, "PAN": None}
    # Update the entities dictionary with recognized entities from the text
    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_] = ent.text
    return entities

if __name__ == '__main__':
    app.run(debug=True,port=5010)
