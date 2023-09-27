from shapely.geometry import box
import shapely
from shapely.ops import cascaded_union


def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1 (tuple): The coordinates of the first bounding box in the format (x_min, y_min, x_max, y_max).
        box2 (tuple): The coordinates of the second bounding box in the format (x_min, y_min, x_max, y_max).

    Returns:
        float: The IoU value, representing the overlap between the two bounding boxes.

    Raises:
        shapely.geos.TopologicalError: If there is an error during the calculation.

    Example:
        box1 = (0, 0, 2, 2)
        box2 = (1, 1, 3, 3)
        iou = calculate_iou(box1, box2)
        print(iou)
        # Output: 0.25
    """
    #     box1 = box(*box1)
    #     box2 = box(*box2)

    if not box1.intersects(box2):
        return 0

    try:
        intersection_area = box1.intersection(box2).area
        union_area = box1.area + box2.area - intersection_area
        return intersection_area / union_area
    except shapely.geos.TopologicalError:
        print("shapely.geos.TopologicalError occurred, iou set to 0")
        return 0


def check_difference(num1, num2, difference=30):
    return -difference <= num1 - num2 <= difference


#     (733 302, 733 404, 406 404, 406 302, 733 302)
def check_x_axis_intersection(box1, box2):
    # Unpacking box coordinates
    x11, y11, x12, y12 = box1
    x21, y21, x22, y22 = box2

    # Check for x axis intersection
    return x21 < x12 and (check_difference(y22, y12) and check_difference(y11, y21))


def check_if_too_close(box1, box2):
    x11, y11, x12, y12 = box1
    x21, y21, x22, y22 = box2

    return (
        check_difference(y11, y21)
        or check_difference(y12, y22)
        and check_difference(x11, x21)
        or check_difference(x12, x22)
    )


def merge_boxes(bboxes):
    """
    Merge overlapping polygons until no more merging is possible.

    Args:
        bboxes (list): A list of bboxes represented as lists of four coordinates [x1, y1, x2, y2].

    Returns:
        list: The final list of merged polygons.
    """
    box_objs = [box(*bbox) for bbox in bboxes]
    merged_boxes = []

    while box_objs:
        union_poly = box_objs.pop(0)
        overlap_boxes = []

        for other_box in box_objs:
            iou = calculate_iou(union_poly, other_box)
            # Checking intersection here
            #             if other_box.intersects(union_poly):
            # Check if only X axis intersects
            if (
                check_x_axis_intersection(union_poly.bounds, other_box.bounds)
                or other_box.within(union_poly)
                or union_poly.within(other_box)
                or iou > 0.1
            ):
                union_poly = cascaded_union([union_poly, other_box])
            else:
                overlap_boxes.append(other_box)

        box_objs = overlap_boxes
        merged_boxes.append(list(union_poly.bounds))

    return merged_boxes


def merge_bbox_recursively(bboxes):
    """
    Merges overlapping bounding boxes recursively until no more merging is possible.

    Args:
        bboxes (list): A list of bounding boxes represented as lists of four coordinates [x1, y1, x2, y2].

    Returns:
        list: The final list of merged bounding boxes.
    """

    bboxes_processed = []

    while True:
        bboxes_processed = merge_boxes(bboxes)

        if len(bboxes_processed) < len(bboxes):
            bboxes = bboxes_processed
        else:
            print("Completely Merged")
            break

    return bboxes_processed


def merging_recursively(bboxes):
    bboxes_processed = []
    while True:
        bboxes_processed = merge_boxes(bboxes)
        if len(bboxes_processed) < len(bboxes):
            print("Merging again", len(bboxes_processed), len(bboxes))
            bboxes = bboxes_processed
        else:
            print("Completely Merged")
            break
    return bboxes_processed


# merged_bbox = merging_recursively(bboxes)
