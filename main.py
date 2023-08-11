import os
import re

from cropt_img import get_and_save_croped_images
from text_detector import detector
from user_manual_search import create_parser

folder_path = os.getcwd()
croped_file_path = os.listdir(f"{folder_path}/crop")


def extract_sort_key(filename):
    # Extract numeric parts using regular expression
    parts = re.findall(r"\d+", filename)
    return [int(part) for part in parts]


def get_list_of_croped_images():

    slide_list = []
    for file_name in croped_file_path:
        slide_list.append(file_name)
    return slide_list


def main():
    """
    Main function to read and parse a PDF user manual.
    """

    pdf_path = "/Users/sohaib/Documents/Tesseract/User manual_Aion S.pdf"
    parser = create_parser(pdf_path)
    print(
        "read all pdfs slide and save images into folder separate by slide number ..........."
    )
    print("Reading and parsing PDF...")
    status = parser.parse_manual()
    if status:
        print("Detecting text in images...")
        detector()
        print("Croping images...using their x,y,w,h coordinates ...............")
        get_and_save_croped_images()

    # print(get_list_of_croped_images())

    import glob

    images = glob.glob("crop/*.png")
    for image in images:
        text = parser.extract_text_from_croped_image(image)
        print(f"Image: {image} \n Text: {text}")


if __name__ == "__main__":
    main()
