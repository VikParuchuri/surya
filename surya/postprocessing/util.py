import math
import copy


def get_line_angle(x1, y1, x2, y2):
    slope = (y2 - y1) / (x2 - x1)

    angle_radians = math.atan(slope)
    angle_degrees = math.degrees(angle_radians)

    return angle_degrees


def rescale_bbox(bbox, processor_size, image_size):
    page_width, page_height = processor_size

    img_width, img_height = image_size
    width_scaler = img_width / page_width
    height_scaler = img_height / page_height

    new_bbox = copy.deepcopy(bbox)
    new_bbox[0] = int(new_bbox[0] * width_scaler)
    new_bbox[1] = int(new_bbox[1] * height_scaler)
    new_bbox[2] = int(new_bbox[2] * width_scaler)
    new_bbox[3] = int(new_bbox[3] * height_scaler)
    return new_bbox


def rescale_point(point, processor_size, image_size):
    # Point is in x, y format
    page_width, page_height = processor_size

    img_width, img_height = image_size
    width_scaler = img_width / page_width
    height_scaler = img_height / page_height

    new_point = copy.deepcopy(point)
    new_point[0] = int(new_point[0] * width_scaler)
    new_point[1] = int(new_point[1] * height_scaler)
    return new_point


def rescale_points(points, processor_size, image_size):
    return [rescale_point(point, processor_size, image_size) for point in points]