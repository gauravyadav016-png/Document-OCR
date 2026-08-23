from flask import Flask, render_template, request, jsonify
import os
from PIL import Image
from PIL.ExifTags import TAGS
import exifread
import numpy as np
from scipy.ndimage import laplace
import torch
from piq import brisque
import os
import json

app = Flask(__name__)

# Define and set the upload folder for the Flask app
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create the directory if it doesn't exist
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to extract EXIF data using Pillow
def get_basic_metadata(image_path):
    image = Image.open(image_path)
    metadata = image._getexif()
    metadata_dict = {}

    if metadata:
        for tag, value in metadata.items():
            tag_name = TAGS.get(tag, tag)
            if isinstance(value, bytes):
                value = value.decode(errors="ignore")  # Convert bytes to string, ignoring errors
            metadata_dict[tag_name] = str(value)  # Convert all values to string
    else:
        metadata_dict['error'] = 'No EXIF metadata found'

    return metadata_dict

def get_detailed_metadata(image_path):
    with open(image_path, 'rb') as f:
        tags = exifread.process_file(f)
        detailed_metadata_dict = {}
        for tag in tags.keys():
            if tag not in ('JPEGThumbnail', 'TIFFThumbnail', 'Filename', 'EXIF MakerNote'):
                value = tags[tag]
                if isinstance(value, bytes):
                    value = value.decode(errors="ignore")
                detailed_metadata_dict[tag] = str(value)

    return detailed_metadata_dict
                
def normalize_value(value, min_value, max_value):
    # Scale the value from [min_value, max_value] to [0, 100]
    normalized_value = (value - min_value) / (max_value - min_value) * 100
    return np.clip(normalized_value, 0, 100)

def calculate_normalized_snr(image_path, min_snr, max_snr):
    image = Image.open(image_path)
    image = image.convert('L')  # Convert to grayscale
    image_data = np.array(image)
    signal = np.mean(image_data)
    noise = np.std(image_data)
    snr = signal / noise if noise != 0 else 0
    return normalize_value(snr, min_snr, max_snr)

def calculate_normalized_sharpness(image_path, min_sharpness, max_sharpness):
    image = Image.open(image_path)
    image = image.convert('L')  # Convert to grayscale
    image_data = np.array(image)
    edge_amplitude = np.abs(laplace(image_data)).mean()  # Use Laplacian operator to find edges
    return normalize_value(edge_amplitude, min_sharpness, max_sharpness)

# def assess_image_quality(image_path, imagepath2):
#     # Load the image
#     image = Image.open(image_path).convert('RGB')
#     image = np.array(image)
#     # Convert image to PyTorch tensor
#     tensor1 = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    
#     image2 = Image.open(imagepath2).convert('RGB')
#     image2 = np.array(image2)
#     tensor2 = torch.from_numpy(image2).permute(2, 0, 1).unsqueeze(0).float() / 255.0
#     # Compute BRISQUE score
#     brisque_score = 100-brisque(tensor1)
#     # Compute PSNR (Peak Signal-to-Noise Ratio)
#     # For PSNR, you need a reference and a distorted image. Here we use the same image as an example.
#     psnr_score = psnr(tensor1, tensor2)
#     # Compute SSIM (Structural Similarity Index)
#     ssim_score = ssim(tensor1, tensor2)
#     # Compute VSI (Visual Saliency-induced Index)
#     vsi_score = vsi(tensor1, tensor2)

#     return {
#         "BRISQUE": brisque_score.item(),
#         "PSNR": psnr_score.item(),
#         "SSIM": ssim_score.item(),
#         "VSI": vsi_score.item()
#     }
def assess_image_quality_no_reference(image_path):
    # Load the image
    image = Image.open(image_path).convert('RGB')
    image = np.array(image)
    # Convert image to PyTorch tensor
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    # Compute BRISQUE score
    brisque_score = 100-brisque(tensor)
    return {
        "BRISQUE Score/Accuracy": brisque_score.item()
    }

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify(error='No file part')
        file = request.files['file']
        if file.filename == '':
            return jsonify(error='No selected file')
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Process the file based on the selected option
            output = ""
            if 'basic' in request.form:
                metadata = get_basic_metadata(filepath)
                output = json.dumps(metadata, indent=4)
            elif 'detailed' in request.form:
                detailed_metadata = get_detailed_metadata(filepath)
                output = json.dumps(detailed_metadata, indent=4)
            elif 'quality' in request.form:
                quality_scores = assess_image_quality_no_reference(filepath)
                output = json.dumps(quality_scores, indent=4)

            # Delete the uploaded file after processing
            os.remove(filepath)

            return f'<pre>{output}</pre>'

    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit name=basic value='Get Basic Details'>
      <input type=submit name=detailed value='Get Detailed EXIF'>
      <input type=submit name=quality value='Get Quality Score'>
    </form>
    '''
if __name__ == '__main__':
   app.run(debug=True,port=5004)