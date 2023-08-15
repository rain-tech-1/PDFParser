import json
import os
import re

import cv2
import imgproc
import numpy as np
import pytesseract
from dotenv import load_dotenv
from pdf2image import convert_from_path
from utils import NumpyEncoder
from text_detector.imgproc import PIL2array
from text_detector.text_detector import detector, load_default_model
from BERT_Embeddings.embedings import get_embeddings
from translation.chinese_to_english import translate

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
    name = re.findall(r"pdf/(.*).pdf", pdf_path)[0]
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
                cropped_image = image[y1:y2, x1:x2]

                # Extract text from croped images
                text = pytesseract.image_to_string(
                    cropped_image, lang=TESSERACT_LANG
                ).strip("\n")

                # Translate to english
                translated_text = translate(text)
                quantised_ch_embeddings, normal_ch_embeddings = get_embeddings(
                    text, "ch"
                )
                quantised_en_embeddings, normal_en_embeddings = get_embeddings(
                    translated_text, "ch"
                )
                print("Extracted text: ", TESSERACT_LANG)
                result = {
                    "id": f"{name}__{page_number}",
                    "display": "Picture cloud storage Path",
                    "bbox": [str(i) for i in [x1, y1, x2, y2]],
                    "text": text,
                    "text_en": translated_text,
                    "score_text": str(score_text),
                    "text_ch_bert": normal_ch_embeddings,
                    "text_ch_bert_qq": quantised_ch_embeddings,
                    "text_en_bert": normal_en_embeddings,
                    "text_en_bert_qq": quantised_en_embeddings,
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
                json.dump(results, f, indent=4, cls=NumpyEncoder)


if __name__ == "__main__":
    main()
