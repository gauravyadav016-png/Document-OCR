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

def aadhar_mask_and_ocr(thres_image, resized_image):
    reader = easyocr.Reader(['en'])
    result = reader.readtext(thres_image)
    final_image = resized_image.copy()
    number_pattern = re.compile(r"\d{4}")
    
    UID = []
    aadhar_parts = []
    for item in result:
        if len(item) == 3:
            (bbox, text, prob) = item
            if prob > 0.5 and number_pattern.match(text):
                aadhar_parts.append((bbox, text))
    
    # Group the Aadhar number parts based on their vertical alignment
    aadhar_groups = []
    for i, item in enumerate(aadhar_parts):
        if len(item) == 2:
            (bbox1, text1) = item
            grouped = False
            for group in aadhar_groups:
                last_part_bbox = group[-1][0]
                y1, y2 = last_part_bbox[1][1], last_part_bbox[2][1]  # Get the y-coordinates of the last part in the group
                ty1, ty2 = bbox1[1][1], bbox1[2][1]  # Get the y-coordinates of the current part
                if abs(y1 - ty1) < 20 and abs(y2 - ty2) < 20:  # Adjust the threshold as needed
                    group.append((bbox1, text1))
                    grouped = True
                    break
            if not grouped:
                aadhar_groups.append([(bbox1, text1)])
    
    # Mask the first 8 digits of each Aadhar number
    for group in aadhar_groups:
        if len(group) == 3:  # Check if the group has all 3 parts of an Aadhar number
            aadhar_text = ''.join([text for _, text in group])
            UID.append(aadhar_text)
            for i, item in enumerate(group):
                if len(item) == 2:
                    (bbox, text) = item
                    (tx1, ty1, tx2, ty2) = bbox[0][0], bbox[0][1], bbox[2][0], bbox[2][1]  # Update the unpacking
                    if i < 2:  # Mask only the first 2 parts (8 digits)
                        final_image = cv2.rectangle(final_image, (tx1, ty1), (tx2, ty2), (255, 255, 255), -1)
    
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
    app.run(debug=True,port=5005)