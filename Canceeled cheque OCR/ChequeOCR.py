from flask import Flask, request, jsonify, render_template_string
from werkzeug.utils import secure_filename
import os
import pytesseract
from PIL import Image, ExifTags
import cv2
import numpy as np
import easyocr
import re
from keras.applications import xception
from keras.preprocessing import image
import numpy
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D, Dense

app = Flask(__name__)

# Configuration for file upload
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'tif', 'tiff'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Check if the upload folder exists, create if not
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    
def build_image_categorization_model():
    model = Sequential()
    model.add(GlobalAveragePooling2D(input_shape=(7, 7, 2048)))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    return model


base_model_Xception = xception.Xception(weights='xception_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False)
def load_build_image_categorization_model(model_file):
    model = build_image_categorization_model()
    model.load_weights(model_file)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

# Build and load the cheque detection model
model = load_build_image_categorization_model('cheque.h5py')

def cheque_tagging(input_file):
    img = image.load_img(input_file, target_size=(224, 224))
    x = image.img_to_array(img)
    x = xception.preprocess_input(x)
    x = numpy.array([x])
    x = base_model_Xception.predict(x)
    y_score = model.predict(x)
    prediction = numpy.argmax(y_score)
    score = numpy.max(y_score)
    if prediction > 0:
        return 'cheque', score
    else:
        return 'non_cheque', score
# Function to check allowed file types
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
           
def perform_ocr(image_path, output_txt_path):
    # Load the image
    image = Image.open(image_path)
    reader = easyocr.Reader(['en'])
    
    # Determine the size of the image
    width, height = image.size

    # Define the portion of the image to keep (e.g., top 80%)
    # Adjust the 0.8 value as needed to exclude the MRZ part
    top_portion_height = int(height * 0.8)

    # Crop the image to exclude the MRZ
    cropped_image = image.crop((0, 0, width, top_portion_height))

    # Optionally, if you want to save or process the MRZ part separately
    mrz_part = image.crop((0, top_portion_height, width, height))
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

    
    # rotation_angle=50
    # rotated_image = image.rotate(rotation_angle, expand=1)
    image_cv = np.array(cropped_image)
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    # median = cv2.medianBlur(gray, 3)
    # binary = cv2.adaptiveThreshold(median, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                                cv2.THRESH_BINARY_INV, 11, 2)


    # Histogram equalization to improve the contrast
    # equalized = cv2.equalizeHist(median)  worsing the image  
    bg=cv2.morphologyEx( gray ,cv2.MORPH_DILATE, rect_kernel)
    out_gray=cv2.divide( gray, bg, scale=255)
    # out_binary=cv2.threshold(out_gray, 0, 255, cv2.THRESH_OTSU )[1]
    morimg=cv2.morphologyEx( gray, cv2.MORPH_BLACKHAT, rect_kernel)
    blur=cv2.GaussianBlur(out_gray, (3, 3), 0)
    new_min, new_max = 0, 255
    contrast_stretched = cv2.normalize(out_gray, None, alpha=new_min, beta=new_max, norm_type=cv2.NORM_MINMAX)
    
    
    # Define a small structuring element for opening
    kernel_size = (1, 2)  # Adjust the size as needed
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    morphed = cv2.morphologyEx(contrast_stretched, cv2.MORPH_CLOSE, kernel)
    # Apply opening to separate the words
    opened = cv2.morphologyEx(morphed, cv2.MORPH_OPEN, kernel)

    # Define erosion kernel size and apply erosion
    erosion_size = (1, 2)  # Adjust the size as needed for your specific image
    erosion_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, erosion_size)
    eroded = cv2.erode(opened, erosion_kernel, iterations=1)
    # Perform OCR using pytesseract
    # extracted_text = pytesseract.image_to_string(gray,config = r'--oem 3 --psm 6')
    results = reader.readtext(eroded)
    extracted_text = "\n".join([result[1] for result in results])
    print(extracted_text)
    # cv2.imwrite('a.png',out_gray)
    # cv2.imwrite('b.png',blur)
    # cv2.imwrite('c.png',contrast_stretched)
    # cv2.imwrite('d.png',eroded)
    # cv2.imwrite('e.png',opened)
    # Save the extracted text to a .txt file
    with open(output_txt_path, 'w') as file:
        file.write(extracted_text)
        
    return extracted_text
def find_ifsc_code(ocr_text):
    # Regular expression for IFSC code: 4 letters followed by 0 or O, then 6 alphanumeric characters
    ifsc_pattern = r'[A-Z]{4}[0O][A-Z0-9]{6}'
    matches = re.findall(ifsc_pattern, ocr_text)

    # Correcting the IFSC codes by replacing 'O' with '0' in the alphanumeric part
    corrected_ifsc_codes = []
    for code in matches:
        # Ensure the 5th character is '0'
        if code[4] == 'O':
            code = code[:4] + '0' + code[5:]

        # Replace 'O' with '0' in the alphanumeric part of the code
        corrected_code = ''.join(['0' if ch == 'O' else ch for ch in code[5:]])
        corrected_ifsc_codes.append(code[:5] + corrected_code)

    print(corrected_ifsc_codes)
    return corrected_ifsc_codes

def find_account_number(ocr_text, ifsc_codes):
    # Standard account number pattern
    account_number_pattern = r'\b\d{9,18}\b'

    # Find potential account numbers
    potential_account_numbers = re.findall(account_number_pattern, ocr_text)
    print("Potential Account Numbers:", potential_account_numbers)

    # Filter based on IFSC code's last 4 digits if IFSC code is available
    if ifsc_codes:
        # Extract last four digits of the IFSC code
        last_four_ifsc = [ifsc[-4:] for ifsc in ifsc_codes]

        # Filter account numbers where the first four digits match with last four of IFSC
        account_numbers = [num for num in potential_account_numbers if num[:4] in last_four_ifsc]
    else:
        account_numbers = potential_account_numbers

    return account_numbers


def find_account_number2(ocr_text):
    # Regular expression to match two types of account number patterns
    # 1. Digits separated by spaces: \b\d(?:\s\d){7,15}\b
    # 2. Continuous digits: \b\d{8,16}\b
    account_number_pattern = r'(\b\d(?:\s\d){7,15}\b|\b\d{8,16}\b)'
    matches = re.findall(account_number_pattern, ocr_text)

    # Cleaning and consolidating the matches
    cleaned_matches = []
    for match in matches:
        # Remove spaces and validate length
        cleaned_match = ''.join(match.split())
        if 8 <= len(cleaned_match) <= 16:
            cleaned_matches.append(cleaned_match)

    return cleaned_matches

def is_cancelled_cheque(ocr_text):
    return "CANCELLED" in ocr_text.upper()

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({"error": "No file part"})

        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "No selected file"})

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            
            # Use the cheque_tagging function
            tag, score = cheque_tagging(file_path)
            if tag == 'cheque':
                # Proceed with OCR if it's a cheque
                ocr_text = perform_ocr(file_path, "output.txt")
                ifsc_codes = find_ifsc_code(ocr_text)

                # Logic to find account numbers
                account_numbers = None
                if ifsc_codes:
                    account_numbers = find_account_number(ocr_text, ifsc_codes)
                if not account_numbers:  # If no account numbers found with IFSC, try the other method
                    account_numbers = find_account_number2(ocr_text)

                # Prepare the response
                response = {}
                score=score*100
                response = {"cheque": True, "Accuracy score ": float(score)}
                if ifsc_codes:
                    response['IFSC Codes'] = ifsc_codes[0]
                if account_numbers:
                    response['Account Numbers'] = account_numbers[0]
                if not ifsc_codes and not account_numbers:
                    response['error'] = 'No account numbers or IFSC codes found.'
                print(response)
                
                return jsonify(response)
                # ... rest of your existing OCR logic ...
            else:
                score=1-score
                return jsonify({"cheque": False, "score": float(score)})

    # HTML form for file upload
    return render_template_string('''
        <!doctype html>
        <title>Upload Cheque Image</title>
        <h1>Upload Cheque Image</h1>
        <form method=post enctype=multipart/form-data>
          <input type=file name=file>
          <input type=submit value=Upload>
        </form>
        ''')

if __name__ == '__main__':
    app.run(debug=True,port=5004)
