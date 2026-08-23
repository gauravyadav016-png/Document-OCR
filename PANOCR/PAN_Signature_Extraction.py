from flask import Flask, request, send_file
import cv2
import matplotlib.pyplot as plt
from skimage import measure, morphology
from skimage.color import label2rgb
from skimage.measure import regionprops
import numpy as np
import os
import datetime

app = Flask(__name__)

# Ensure there's a folder to save the uploaded and processed images
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(PROCESSED_FOLDER):
    os.makedirs(PROCESSED_FOLDER)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Save the uploaded image
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            filename = file.filename
            save_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(save_path)
            processed_filename = process_image(save_path, timestamp)
            processed_file_url = f'/static/processed/{processed_filename}'

            return f'''
            <!doctype html>
            <title>Processed Image</title>
            <h1>Processed Image</h1>
            <img src="{processed_file_url}" alt="Processed Image" style="max-width: 500px;">
            <br>
            <a href="/">Upload another image</a>
            '''
    return '''
    <!doctype html>
    <title>Upload an Image</title>
    <h1>Upload an Image</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

def process_image(file_path, timestamp):
    # Parameters to remove small and large size connected pixels outliers
    constant_parameter_1 = 84
    constant_parameter_2 = 250
    constant_parameter_3 = 100
    constant_parameter_4 = 18

    # Read the input image
    img = cv2.imread(file_path, 0)
    img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]  # ensure binary

    # Connected component analysis
    blobs = img > img.mean()
    blobs_labels = measure.label(blobs, background=1)
    image_label_overlay = label2rgb(blobs_labels, image=img)

    the_biggest_component = 0
    total_area = 0
    counter = 0
    for region in regionprops(blobs_labels):
        if region.area > 10:
            total_area += region.area
            counter += 1
        if region.area >= 250 and region.area > the_biggest_component:
            the_biggest_component = region.area

    average = total_area / counter if counter > 0 else 0
    a4_small_size_outliar_constant = ((average / constant_parameter_1) * constant_parameter_2) + constant_parameter_3
    a4_big_size_outliar_constant = a4_small_size_outliar_constant * constant_parameter_4

    # Remove small objects
    pre_version = morphology.remove_small_objects(blobs_labels, a4_small_size_outliar_constant)
    # Remove large objects
    component_sizes = np.bincount(pre_version.ravel())
    too_big = component_sizes > a4_big_size_outliar_constant
    too_big_mask = too_big[pre_version]
    pre_version[too_big_mask] = 0

    # Save and process the image to extract the signature
    plt.imsave('pre_version.png', pre_version)
    img = cv2.imread('pre_version.png', 0)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # Save the processed image with the timestamp
    processed_image_name = f'{os.path.splitext(os.path.basename(file_path))[0]}_{timestamp}.png'
    processed_image_path = os.path.join(PROCESSED_FOLDER, processed_image_name)
    cv2.imwrite(processed_image_path, img)

    return processed_image_name
    

if __name__ == '__main__':
    app.run(debug=True,port=5007)