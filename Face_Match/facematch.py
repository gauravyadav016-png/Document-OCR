from flask import Flask, jsonify, request, redirect
from PIL import Image
import face_recognition
import numpy as np
import math
app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'pan_card' not in request.files or 'aadhaar_card' not in request.files:
            return redirect(request.url)

        pan_card = request.files['pan_card']
        aadhaar_card = request.files['aadhaar_card']

        if pan_card.filename == '' or aadhaar_card.filename == '':
            return jsonify({"Status": "Error", "Response": "Please upload both card images"})

        if allowed_file(pan_card.filename) and allowed_file(aadhaar_card.filename):
            return jsonify(extract_and_compare_faces(pan_card, aadhaar_card))
        else:
            return jsonify({"Status": "Error", "Response": "Invalid File Format"})

    return '''
    <!doctype html>
    <title>Face Matching</title>
    <h1>Upload PAN and Aadhaar Card Images</h1>
    <form method="POST" action="/" enctype="multipart/form-data">
     <h3>Upload PAN Card</h3> <input type="file" name="pan_card">
     <h3>Upload Aadhaar Card</h3> <input type="file" name="aadhaar_card">
     <input type="submit" value="Upload">
    </form>
    '''

def extract_and_compare_faces(pan_card_stream, aadhaar_card_stream):
    # Function to resize images for efficient processing
    def resize_image(image_stream, max_size=800):
        image = Image.open(image_stream)
        image.thumbnail((max_size, max_size))
        return np.array(image)

    # Load and resize the PAN card image, then find face locations
    pan_image = resize_image(pan_card_stream)
    print("2")
    pan_faces = face_recognition.face_locations(np.array(pan_image), model="cnn")
    print("a")
    # Load and resize the Aadhaar card image, then find face locations
    aadhaar_image = resize_image(aadhaar_card_stream)
    print("b")
    aadhaar_faces = face_recognition.face_locations(np.array(aadhaar_image), model="cnn")
    print("3")
    # Check if faces are found in both images
    if not pan_faces and not aadhaar_faces:
        return {"Status": "Error", "Response": "No face found in both of the images"}
    
    if not pan_faces:
        return {"Status": "Error", "Response": "No face found in PAN images"}
    
    if  not aadhaar_faces:
        return {"Status": "Error", "Response": "No face found in Aadhaar images"}

    # Extract the first face encoding found in both images
    pan_face_encoding = face_recognition.face_encodings(pan_image, known_face_locations=pan_faces)[0]
    print("4")
    aadhaar_face_encoding = face_recognition.face_encodings(aadhaar_image, known_face_locations=aadhaar_faces)[0]
    print("5")
    # Compare faces and calculate face distance
    match = face_recognition.compare_faces([pan_face_encoding], aadhaar_face_encoding, tolerance=0.5)
    print("6")
    # Calculate matching percentage
    # In your extract_and_compare_faces function
    face_distance = face_recognition.face_distance([pan_face_encoding], aadhaar_face_encoding)[0]
    confidence_score = face_distance_to_conf(face_distance) * 100  # Convert to percentage

    return {
        "match": match[0].item(),
        "face_distance": face_distance,
        "matching_percentage": confidence_score
    }
    
def face_distance_to_conf(face_distance, face_match_threshold=0.6):
    if face_distance > face_match_threshold:
        range1 = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (range1 * 2.0)
        return linear_val
    else:
        range1 = face_match_threshold
        linear_val = 1.0 - (face_distance / (range1 * 2.0))
        return linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))

if __name__ == "__main__":
    app.run( port=5003, debug=True)
