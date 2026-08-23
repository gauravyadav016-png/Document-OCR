import time
import uuid
from flask import Flask, request, send_file, render_template_string
import cv2
import pytesseract
import re
from scipy import ndimage
import os

app = Flask(__name__)
# To detect orientation and perform rotation
def rotate(image, center = None, scale = 1.0):
    try:
        angle = int(re.search('(?<=Rotate: )\d+', pytesseract.image_to_osd(image)).group(0))
    except Exception as e:
        print(f"Error: {e}")
        return image 
    # angle=int(re.search('(?<=Rotate: )\d+', pytesseract.image_to_osd(image)).group(0))
    (h, w) = image.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    rotated = ndimage.rotate(image, float(angle) * -1)
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


# To perform preprocessing before OCR
# def preprocessing(image):
#     w, h = image.shape[0],image.shape[1]
#     if w < h:
#         image = rotate(image)
#     resized_image = cv2.resize(image, None, fx=3, fy=3, interpolation=cv2.INTER_LINEAR)
#     grey_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
#     blur_image = cv2.medianBlur(grey_image, 3)
#     thres_image = cv2.adaptiveThreshold(blur_image,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
#                 cv2.THRESH_BINARY,13,7)
#     return thres_image, resized_image

# To perform OCR and mask first 8 digits
def aadhar_mask_and_ocr(thres_image, resized_image):
    custom_config = r'--oem 3 --psm 6'
    try:
        d = pytesseract.image_to_data(thres_image, output_type=pytesseract.Output.DICT, config=custom_config)
    except pytesseract.TesseractError as e:
        print(f'Error: {e}')
        return None, None  # or however you want to handle the error
    
    # d = pytesseract.image_to_data(thres_image, output_type=pytesseract.Output.DICT)
    keys = list(d.keys())
    number_pattern = r"(?<!\d)\d{4}(?!\d)"
    n_boxes = len(d['text'])
    c = 0
    temp = []
    UID = []
    final_image = resized_image.copy()
    for i in range(n_boxes):
        if int(d['conf'][i]) > 20:
            if re.match(number_pattern, d['text'][i]):
                if c<2:
                    (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                    final_image = cv2.rectangle(resized_image, (x, y), (x + w, y + h), (255, 255, 255), -1)
                    temp.append(d['text'][i])
                    c+=1
                elif (c>=2) and (d['text'][i] in temp):
                    (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                    final_image = cv2.rectangle(resized_image, (x, y), (x + w, y + h), (255, 255, 255), -1)
                elif c==2:
                    UID = temp + [ d['text'][i]]
                    c+=1
                else:
                    continue
    final_image = cv2.resize(final_image, None, fx=0.33, fy=0.33)
    return final_image, UID

# image_path = 'images/img1.jpeg'  # Specify your image path here
# image = cv2.imread(image_path)

# thres_image, resized_image = preprocessing(image)
# masked_image, UID = aadhar_mask_and_ocr(thres_image, resized_image)

# print('Masked digits in given image. Displaying...')
# cv2.imshow('amask' + image_path, masked_image)

# # To save output
# output_path = 'amasked_image.jpeg'  # Specify your desired output path here
# cv2.imwrite(output_path, masked_image)

# print('UID' + ' : ' + ' '.join(UID))
# print('Press q over output window to close')
# cv2.waitKey(0)
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