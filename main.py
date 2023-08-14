import json
import os
import re

import cv2
import imgproc
import numpy as np
import pytesseract
from dotenv import load_dotenv
from pdf2image import convert_from_path

from text_detector.imgproc import PIL2array
from text_detector.text_detector import detector, load_default_model

load_dotenv()
TESSERACT_LANG = os.getenv("TESSERACT_LANG", "eng")
# Loading text detector model
load_default_model()


def main(pdf_path="pdf/User manual_Aion S.pdf", save_results="test/"):
    """
    Main function to read and parse a PDF user manual.
    """
    global TESSERACT_LANG

    # Parsing PDF
    images = convert_from_path(pdf_path)
    print(images[1], type(images[1]), len(images))

    if images:
        for page_number, image in enumerate(images[:2]):
            results = []
            print("Detecting text in images...")
            image = PIL2array(image)
            bboxes, polys, score_texts = detector(image)
            print("Extracting text from images...")

            # Extracting text from images
            for bbox, score_text in zip(bboxes, score_texts):
                x1, y1 = np.asarray(bbox[0], dtype=np.int32)
                x2, y2 = np.asarray(bbox[2], dtype=np.int32)

                # Crop bounding boxes
                cropped_image = image[y1:y2, x1:x2]

                # Extract text from croped images
                text = pytesseract.image_to_string(cropped_image, lang=TESSERACT_LANG)

                print("Extracted text: ", TESSERACT_LANG)
                result = {
                    "page_number": str(page_number),
                    "bbox": [str(i) for i in [x1, y1, x2, y2]],
                    "text": text,
                    "score_text": str(score_text),
                    "BERT_embedings": "To Do",
                }
                results.append(result)

                # Save cropped images
                if save_results:
                    print("Saving Cropped Images ...")
                    save_cropped_images = (
                        f"{save_results}/page_number_{page_number}/cropped_image/"
                    )
                    if not os.path.exists(save_cropped_images):
                        os.makedirs(save_cropped_images)
                    cv2.imwrite(
                        f"{save_cropped_images}/croped_images_{bbox}.png",
                        cropped_image,
                    )

            # Saving Results in Json
            save_json_at = f"{save_results}/page_number_{page_number}/"
            print("Saving results...")
            if not os.path.exists(save_json_at):
                os.makedirs(save_json_at)
            with open(f"{save_json_at}/results.json", "w") as f:
                json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
