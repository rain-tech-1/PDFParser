from PIL import Image
import os


slide_list = []
folder_path = os.getcwd()


def read_polygons(path):
    coordinates = []

    with open(path, "r") as file:
        for line in file:
            line = line.strip()  # Remove newline characters
            coords = list(map(int, line.split(",")))
            coordinates.append(coords)

    return coordinates


def all_slides():
    # List all files in the folder
    file_names = os.listdir(f"{folder_path}/data")

    # Print the list of file names

    for file_name in file_names:
        slide_list.append(file_name)
    return True


def convert_coordinates(coordinates):
    converted_coordinates = []
    for coord in coordinates:
        x1 = min(coord[::2])  # Get the minimum x value
        y1 = min(coord[1::2])  # Get the minimum y value
        x2 = max(coord[::2])  # Get the maximum x value
        y2 = max(coord[1::2])  # Get the maximum y value

        converted_coordinates.append([x1, y1, x2, y2])
    return converted_coordinates


def crop_image(img, converted_coordinates, slide, folder_path):
    cropped_images = []
    for i, coord in enumerate(converted_coordinates):
        cropped_img = img.crop(coord)
        cropped_images.append(cropped_img)

        # Save the cropped image
        cropped_img.save(f"{folder_path}/crop/page_{slide}_croped_{i+1}.png")


def get_and_save_croped_images():
    status = all_slides()
    if status == True:
        for slide in slide_list:
            slide = slide.split(".")
            slide = slide[0]
            coordinates = read_polygons(f"{folder_path}/result/res_{slide}.txt")
            converted_coordinates = convert_coordinates(coordinates)
            # Open an image file
            img = Image.open(f"{folder_path}/data/{slide}.png")
            crop_image(img, converted_coordinates, slide, folder_path)
