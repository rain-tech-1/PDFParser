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

# Usage
To use this script, run the main.py: just update pdf_path in main.py
