import os
import re

import pytesseract
from fpdf import FPDF
from pdf2image import convert_from_path
from PIL import Image
import json
import pdb
# Set Tesseract data directory
os.environ['TESSDATA_PREFIX'] = '/usr/local/share/'

class PDFParser:
    """
    Class to parse text from scanned PDFs.
    """
    def __init__(self, pdf_path):
        """
        Initialize the PDFParser with the provided PDF file path.

        :param pdf_path: Path to the PDF file.
        """
        self.pdf_path = pdf_path
        self.extracted_text = ""
        self.current_directory = os.getcwd()
    
    def extract_images_from_pdf(self):
        """
        Extract text from scanned PDF using Tesseract OCR.

        :return: Extracted text from the PDF.
        """
        try:
            images = convert_from_path(self.pdf_path)
            for i, image in enumerate(images):
                image.save(f"{self.current_directory}/data/{i+1}.png")
                
        except:
            return False
        
        return True


        return self.extracted_text
    def extract_text_from_croped_image(self,croped_image_path):
        """
        Extract text from scanned PDF using Tesseract OCR.

        :return: Extracted text from the PDF.
        """

        image = Image.open(croped_image_path)
        self.extracted_text = pytesseract.image_to_string(image, lang='tessdata/chi_sim')
        
        data_dict = {"ExtractedText": self.extracted_text}

        # Dump the dictionary into a JSON format
        json_data = json.dumps(data_dict)

        return self.extracted_text

    @staticmethod
    def remove_header_footer(text):
        """
        Remove header and footer text from extracted content.

        :param text: Text extracted from the PDF.
        :return: Text with header and footer removed.
        """
        # TODO: Implement logic to remove header and footer text
        # For demonstration, we'll assume header and footer are enclosed in square brackets.
        header_footer_pattern = r"\[.*?\]"
        cleaned_text = re.sub(header_footer_pattern, "", text)
        return cleaned_text

    def parse_manual(self):
        """
        Parse the user manual content.

        :return: Parsed and cleaned text from the user manual.
        """
        status = self.extract_images_from_pdf()

        return status

def create_parser(pdf_path):
    """
    Create a PDFParser object for the provided PDF path.

    :param pdf_path: Path to the PDF file.
    :return: PDFParser object.
    """
    parser = PDFParser(pdf_path)
    return parser