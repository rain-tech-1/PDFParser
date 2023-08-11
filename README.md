# PDFParser

# User Manual Search

This Python script is used to parse a PDF user manual, extract text from it (even if it's a scanned document), and remove headers and footers. It uses the Tesseract OCR engine for text extraction and PyPDF2 for PDF handling.

# Prerequisites

Before running this script, you need to install some dependencies:

Python 3.10 or higher
`brew install python`

install conda and make environment

`conda create --name myenv python=3.11`
`conda activate myenv`

install requirements
`pip install -r requirements.txt`

Tesseract: An OCR engine.
`brew install tesseract`

Add Tesseract to your PATH:
`export PATH=$PATH:/path/to/tesseract`

Poppler: A PDF rendering library.
`brew install poppler`

For Chinese language support, download the trained data file for Simplified Chinese:

`sudo mkdir -p /usr/local/share/tessdata/`
`sudo curl -L -o /usr/local/share/tessdata/chi_sim.traineddata https://github.com/tesseract-ocr/tessdata/raw/main/chi_sim.traineddata`

if text detector model not found, download it from
`sudo wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/1n/p')&id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ" -O craft_mlt_25k.pth && rm -rf /tmp/cookies.txt`

run test detector scrip 
`python text_detector.py --trained_model=craft_mlt_25k.pth --source_data=path-of-images-folder/`  

# Usage

To use this script, run the main.py: just update pdf_path in main.py
