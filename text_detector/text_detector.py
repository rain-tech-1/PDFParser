"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

import os
import time
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from dotenv import load_dotenv

# -*- coding: utf-8 -*-

# from shapely.ops import unary_union
from torch.autograd import Variable
from shapely.geometry import Polygon
import text_detector.craft_utils as craft_utils
import text_detector.imgproc as imgproc
from text_detector.craft import CRAFT
from text_detector.bbox_merger import merging_recursively

load_dotenv()
net = None
refine_net = None
cuda = False if "0" == os.getenv("CUDA", "0") else True
trained_model = os.getenv("TEXT_DETECTION_MODEL_PATH", "weights/craft_mlt_25k.pth")

print("CUDA is set to: ", cuda)


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")


def test_net(
    net,
    image,
    text_threshold,
    link_threshold,
    low_text,
    cuda,
    poly,
    canvas_size,
    mag_ratio,
    show_time,
    refine_net=None,
):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(
        image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio
    )
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0, :, :, 0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(
        score_text, score_link, text_threshold, link_threshold, low_text, poly
    )

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None:
            polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    if show_time:
        print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text


def LoadModel(
    trained_model="weights/craft_mlt_25k.pth",
    cuda=False,
    refine=False,
    refiner_model="weights/craft_refiner_CTW1500.pth",
):
    # load net
    net = CRAFT()  # initialize
    print("Loading weights from checkpoint (" + trained_model + ")")
    if cuda:
        net.load_state_dict(copyStateDict(torch.load(trained_model)))

    else:
        net.load_state_dict(
            copyStateDict(torch.load(trained_model, map_location=torch.device("cpu")))
        )

    if cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

    # LinkRefiner
    refine_net = None
    if refine:
        from refinenet import RefineNet

        refine_net = RefineNet()
        print("Loading weights of refiner from checkpoint (" + refiner_model + ")")
        if cuda:
            refine_net.load_state_dict(copyStateDict(torch.load(refiner_model)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(
                copyStateDict(torch.load(refiner_model, map_location="cpu"))
            )

        refine_net.eval()
    return net, refine_net


def load_default_model():
    global net, cuda, trained_model, refine_net
    net = None
    if net is None:
        net, refine_net = LoadModel(
            trained_model=trained_model,
            cuda=cuda,
            refiner_model="weights/craft_mlt_25k.pthh",
        )


def detector(
    # read image with imgproc.LoadImage()
    image: np.asarray = None,
    text_threshold: float = 0.7,
    low_text: float = 0.4,
    link_threshold: float = 0.4,
    canvas_size: int = 1280,
    mag_ratio: float = 1.5,
    poly: bool = False,
    show_time: bool = False,
    model_path: str = "",
):
    global net, cuda, trained_model, refine_net

    if model_path:
        trained_model = model_path
    # Using default trained model if CRAFT not passed
    # or load from file if passed
    if net is None:
        net, refine_net = LoadModel(
            trained_model=trained_model,
            cuda=cuda,
            refiner_model="weights/craft_refiner_CTW1500.pth",
        )

    bboxes, polys, score_text = test_net(
        net,
        image,
        text_threshold,
        link_threshold,
        low_text,
        cuda,
        poly,
        canvas_size,
        mag_ratio,
        show_time,
        refine_net,
    )

    return bboxes, polys, score_text


# def calculate_iou(box1, box2):
#     """
#     Calculate the Intersection over Union (IoU) between two bounding boxes.

#     Args:
#         box1 (tuple): The coordinates of the first bounding box in the format (x_min, y_min, x_max, y_max).
#         box2 (tuple): The coordinates of the second bounding box in the format (x_min, y_min, x_max, y_max).

#     Returns:
#         float: The IoU value, representing the overlap between the two bounding boxes.

#     Raises:
#         shapely.geos.TopologicalError: If there is an error during the calculation.

#     Example:
#         box1 = (0, 0, 2, 2)
#         box2 = (1, 1, 3, 3)
#         iou = calculate_iou(box1, box2)
#         print(iou)
#         # Output: 0.25
#     """
#     box1 = box(*box1)
#     box2 = box(*box2)

#     if not box1.intersects(box2):
#         return 0

#     try:
#         intersection_area = box1.intersection(box2).area
#         union_area = box1.area + box2.area - intersection_area
#         return intersection_area / union_area
#     except shapely.geos.TopologicalError:
#         print("shapely.geos.TopologicalError occurred, iou set to 0")
#         return 0


# def merge_boxes(bboxes):
#     """
#     Merge overlapping polygons until no more merging is possible.

#     Args:
#         bboxes (list): A list of bboxes represented as lists of four coordinates [x1, y1, x2, y2].

#     Returns:
#         list: The final list of merged polygons.
#     """
#     box_objs = [box(*bbox) for bbox in bboxes]
#     merged_boxes = []

#     while box_objs:
#         union_poly = box_objs.pop(0)
#         overlap_boxes = []

#         for other_box in box_objs:
#             if other_box.intersects(union_poly):
#                 union_poly = cascaded_union([union_poly, other_box])
#             else:
#                 overlap_boxes.append(other_box)

#         box_objs = overlap_boxes
#         merged_boxes.append(list(union_poly.bounds))

#     return merged_boxes


# def merge_bbox_recursively(bboxes):
#     """
#     Merges overlapping bounding boxes recursively until no more merging is possible.

#     Args:
#         bboxes (list): A list of bounding boxes represented as lists of four coordinates [x1, y1, x2, y2].

#     Returns:
#         list: The final list of merged bounding boxes.
#     """

#     bboxes_processed = []

#     while True:
#         bboxes_processed = merge_boxes(bboxes)

#         if len(bboxes_processed) < len(bboxes):
#             bboxes = bboxes_processed
#         else:
#             print("Completely Merged")
#             break

#     return bboxes_processed


def predict(image, size=1280):
    all_boxes = []
    height, width, channel = image.shape
    for y in range(0, height + size, size):
        for x in range(0, width + size, size):
            try:
                pred_image = image[y : y + size, x : x + size]

                bboxes = detect_objects(pred_image, size)

                bboxes = [
                    [bbox[0] + x, bbox[1] + y, bbox[2] + x, bbox[3] + y]
                    for bbox in bboxes
                ]

                # Converting into polygons
                x_padding = 50
                y_padding = 0
                bboxes = [
                    [
                        bbox[0] - x_padding,
                        bbox[1] - y_padding,
                        bbox[2] + x_padding,
                        bbox[3] + y_padding,
                    ]
                    for bbox in bboxes
                ]

                all_boxes.extend(bboxes)
            except Exception as e:
                print(e)
                pass
    all_boxes = merging_recursively(all_boxes)
    return all_boxes


# def merging_recursively(bboxes):
#     bboxes_processed = []
#     while True:
#         bboxes_processed = merge_boxes(bboxes)
#         if len(bboxes_processed) < len(bboxes):
#             print("Merging again", len(bboxes_processed), len(bboxes))
#             bboxes = bboxes_processed
#         else:
#             print("Completely Merged")
#             break
#     return bboxes_processed


def detect_objects(image, size):
    bboxes = []

    bboxes, polys, score_texts = detector(
        image,
        model_path="../text_detector/weights/craft_mlt_25k.pth",
        text_threshold=0.3,
        canvas_size=size,
    )

    bboxes = [Polygon(bbox).bounds for bbox in bboxes]

    bboxes = [[int(i) for i in sublist] for sublist in bboxes]
    bboxes = sorted(bboxes, key=lambda x: x[1])

    return bboxes
