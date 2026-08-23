from flask import Flask, request, send_file, render_template_string
import cv2
import easyocr
import re
import os
import time
import uuid

app = Flask(__name__)

def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated
def preprocessing(image):
    # w, h = image.shape[0], image.shape[1]
    # if w < h:
    #     image = rotate(image)
    resized_image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    grey_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    blur_image = cv2.medianBlur(grey_image, 3)
    thres_image = cv2.adaptiveThreshold(blur_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 13, 7)
    return thres_image, resized_image


# To perform OCR and mask all digits
def aadhar_mask_and_ocr(thres_image, resized_image):
    reader = easyocr.Reader(['en'])
    result = reader.readtext(thres_image)
    final_image = resized_image.copy()
    number_pattern = re.compile(r"(?<!\d)\d{4}(?!\d)")
    
    UID = []
    for (bbox, text, prob) in result:
        if prob > 0.5 and number_pattern.match(text):
            x_min, y_min = map(int, bbox[0])
            x_max, y_max = map(int, bbox[2])
            width_per_digit = (x_max - x_min) / 12
            
            # Adjust x_max to cover only the first 8 digits
            x_max = int(x_min + (8 * width_per_digit))
            final_image = cv2.rectangle(final_image, (x_min, y_min), (x_max, y_max), (255, 255, 255), -1)
            
            UID.append(text)

    final_image = cv2.resize(final_image, None, fx=0.33, fy=0.33)
    return final_image, UID



@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            uploads_dir = 'uploads'
            if not os.path.exists(uploads_dir):
                os.makedirs(uploads_dir)
            
            # Generate a unique filename using a timestamp or UUID
            unique_filename = f"{int(time.time())}_{file.filename}"
            file_path = os.path.join(uploads_dir, unique_filename)
            file.save(file_path)
            
            print(f"Processing file: {unique_filename}")  # Debugging
            
            image = cv2.imread(file_path)
            thres_image, resized_image = preprocessing(image)
            masked_image, UID = aadhar_mask_and_ocr(thres_image, resized_image)
            
            if masked_image is None:
                os.remove(file_path)  # Delete the uploaded file after processing
                return 'Error during processing.', 400  # or however you want to handle the error
            
            output_dir = 'output_file'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            output_path = os.path.join(output_dir, f'masked_{unique_filename}')
            cv2.imwrite(output_path, masked_image)
            
            os.remove(file_path)  # Delete the uploaded file after processing
            # print(UID)
            return send_file(output_path, mimetype='image/jpeg')
    
    return render_template_string('''
        <!doctype html>
        <title>Upload Image</title>
        <h1>Upload Image</h1>
        <form method=post enctype=multipart/form-data>
          <input type=file name=file>
          <input type=submit value=Upload>
        </form>
    ''')

@app.after_request
def add_header(response):
    response.cache_control.no_store = True
    response.cache_control.no_cache = True
    response.cache_control.must_revalidate = True
    response.cache_control.proxy_revalidate = True
    response.expires = 0
    response.pragma = 'no-cache'
    return response
if __name__ == '__main__':
    app.run(debug=True,port=5003)