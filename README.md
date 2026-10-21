# In-AE-CD: Document AI & Identity Verification Pipeline

An end-to-end document-processing toolkit that extracts text from Indian identity and financial documents using OCR, verifies identity by matching faces across documents, and checks image quality before processing.

## Features

- **Aadhaar OCR**: extracts text fields from Aadhaar card images.
- **PAN OCR**: extracts text fields from PAN card images.
- **Passport OCR**: extracts text fields from passport images.
- **Cancelled Cheque OCR**: extracts account details from cancelled cheque images.
- **Face Match**: compares faces across PAN and Aadhaar images using CNN-based face detection, face encodings, and a configurable distance threshold.
- **Image Quality Check**: flags blurry, dark, or low-resolution images before OCR runs.
- **Aadhaar Masking** (`aadhar-mask-ocr`): detects and masks Aadhaar numbers in documents to protect sensitive data.

## Project Structure

```
In-AE-CD/
├── Aadhaar_OCR/             # Aadhaar text extraction
├── PANOCR/                  # PAN card text extraction
├── Passport OCR/            # Passport text extraction
├── Canceeled cheque OCR/    # Cancelled cheque text extraction
├── Face_Match/              # Face verification across documents
├── Image Quality Check/     # Pre-OCR image quality checks
├── aadhar-mask-ocr/         # Aadhaar number masking
└── README.md
```

## Tech Stack

- **Language:** Python
- **OCR and image processing:** OpenCV
- **Face verification:** CNN-based face detection and face encodings
- **API layer:** Flask

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
git clone https://github.com/gauravyadav016-png/In-AE-CD.git
cd In-AE-CD
pip install -r requirements.txt
```

### Usage

Run the module you need from its folder. For example:

```bash
cd Face_Match
python face_match.py --doc1 path/to/pan.jpg --doc2 path/to/aadhaar.jpg
```

Adjust file names and arguments to match each module's script.

## Example Workflow

1. Run **Image Quality Check** on the uploaded document.
2. If the image passes, run the matching **OCR** module (Aadhaar, PAN, Passport, or Cheque).
3. Run **Face Match** to compare the face on the PAN and Aadhaar images.
4. Optionally run **Aadhaar Masking** before storing or sharing the document.

## Privacy Note

This project handles sensitive identity documents. Use it only with test data or documents you are authorised to process, and do not commit real identity documents to the repository.

## Future Improvements

- Expose all modules through a single Flask API with a unified endpoint.
- Add automated tests with sample (synthetic) documents.
- Add a `requirements.txt` with pinned versions.

## Author

**Gaurav Yadav**
Email: gauravyadavofficial16@gmail.com
LinkedIn: https://www.linkedin.com/in/gauravyadav016/
GitHub: https://github.com/gauravyadav016-png
