import json
import os
import re
import uuid

import cv2
import imgproc
import numpy as np
import pytesseract
from dotenv import load_dotenv
from pdf2image import convert_from_path

from BERT_Embeddings.embedings import get_embeddings
from text_detector.imgproc import PIL2array
from text_detector.text_detector import predict, load_default_model
from translation.chinese_to_english import translate
from utils import NumpyEncoder, save_image

load_dotenv()
TESSERACT_LANG = os.getenv("TESSERACT_LANG", "eng")
# Loading text detector model
load_default_model()


def text_post_processing(text):
    text = text.replace("\n", " ")
    text = text.replace("\f", " ")
    text = text.replace("\t", " ")
    text = text.replace("\r", " ")
    return text


def main(pdf_path="pdf/MS_SOP16_SOP_20230118.pdf", save_results="test/"):
    """
    Main function to read and parse a PDF user manual.
    """
    global TESSERACT_LANG
    start = 1

    # Parsing PDF
    while True:
        end = start + 10
        images = convert_from_path(
            pdf_path,
            dpi=300,
            fmt="png",
            thread_count=4,
            first_page=start,
            last_page=end,
        )
        name = re.findall(r"pdf/(.*).pdf", pdf_path)[0]
        print(images[1], type(images[1]), len(images))

        if images:
            for page_number, image in enumerate(images):
                results, text_to_ignore = [], []
                print("Detecting text in images...")
                image = PIL2array(image)
                bboxes = predict(image)
                print("Extracting text from images...")
                # Extracting text from headers and footers

                # Extracting text from images
                for bbox in bboxes:
                    print("bbox:", bbox)
                    # convert to integer
                    bbox = [int(i) for i in bbox]
                    x1, y1, x2, y2 = bbox

                    cropped_image = image[y1:y2, x1:x2]

                    # Extract text from croped images
                    text = pytesseract.image_to_string(
                        cropped_image, lang=TESSERACT_LANG
                    ).strip("\n")

                    if len(text) > 0:
                        # Translate to english
                        # translated_text = translate(text)
                        # (
                        #     quantised_ch_embeddings,
                        #     normal_ch_embeddings,
                        #     min_max_array_per_column_ch,
                        # ) = get_embeddings(text, "ch")

                        (
                            quantised_en_embeddings,
                            normal_en_embeddings,
                            min_max_array_per_column_en,
                        ) = get_embeddings(text)

                        # Post processing

                        text = text_post_processing(text)

                        # Only allow numbers and alphabets

                        result = {
                            "id": str(uuid.uuid4()),
                            "display": "Picture cloud storage Path",
                            "bbox": [str(i) for i in [x1, y1, x2, y2]],
                            "text": text,
                            "text_to_ignore": text_to_ignore,
                            # "text_en": translated_text,
                            # "text_ch_bert": normal_ch_embeddings,
                            # "text_ch_bert_qq": quantised_ch_embeddings,
                            "text_en_bert": normal_en_embeddings,
                            "text_en_bert_qq": quantised_en_embeddings,
                            "text_en_bert_qq_min_max": min_max_array_per_column_en,
                            # "text_ch_bert_qq_min_max": min_max_array_per_column_ch,
                        }
                        results.append(result)
                    save_image(save_results, page_number, cropped_image, bbox)
                # Saving Results in Json
                save_json_at = f"{save_results}/page_number_{page_number}/"
                print("Saving results...")
                if not os.path.exists(save_json_at):
                    os.makedirs(save_json_at)
                with open(f"{save_json_at}/results.json", "w") as f:
                    json.dump(results, f, indent=4, cls=NumpyEncoder)
        start += 10


if __name__ == "__main__":
    main()
