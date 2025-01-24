import math
from typing import List

import cv2
import numpy as np

from surya.detection.schema import ColumnLine


def get_line_angle(x1, y1, x2, y2):
    slope = (y2 - y1) / (x2 - x1)

    angle_radians = math.atan(slope)
    angle_degrees = math.degrees(angle_radians)

    return angle_degrees


def get_detected_lines_sobel(image, vertical=True):
    # Apply Sobel operator with a kernel size of 3 to detect vertical edges
    if vertical:
        dx = 1
        dy = 0
    else:
        dx = 0
        dy = 1

    sobelx = cv2.Sobel(image, cv2.CV_32F, dx, dy, ksize=3)


    # Absolute Sobel (to capture both edges)
    abs_sobelx = np.absolute(sobelx)

    # Convert to 8-bit image
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    kernel = np.ones((20, 1), np.uint8)
    eroded = cv2.erode(scaled_sobel, kernel, iterations=1)
    scaled_sobel = cv2.dilate(eroded, kernel, iterations=3)

    return scaled_sobel


def get_detected_lines(image, slope_tol_deg=2, vertical=False, horizontal=False) -> List[ColumnLine]:
    assert not (vertical and horizontal)
    new_image = image.astype(np.float32) * 255  # Convert to 0-255 range
    if vertical or horizontal:
        new_image = get_detected_lines_sobel(new_image, vertical)
    new_image = new_image.astype(np.uint8)

    edges = cv2.Canny(new_image, 150, 200, apertureSize=3)
    if vertical:
        max_gap = 100
        min_length = 10
    else:
        max_gap = 10
        min_length = 4

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=150, minLineLength=min_length, maxLineGap=max_gap)

    line_info = []
    if lines is not None:
        for line in lines:
            vertical_line = False
            horizontal_line = False
            x1, y1, x2, y2 = line[0]
            bbox = [x1, y1, x2, y2]

            if x2 == x1:
                vertical_line = True
            else:
                line_angle = get_line_angle(x1, y1, x2, y2)
                if 90 - slope_tol_deg < line_angle < 90 + slope_tol_deg:
                    vertical_line = True
                elif -90 - slope_tol_deg < line_angle < -90 + slope_tol_deg:
                    vertical_line = True
                elif -slope_tol_deg < line_angle < slope_tol_deg:
                    horizontal_line = True

            if bbox[3] < bbox[1]:
                bbox[1], bbox[3] = bbox[3], bbox[1]
            if bbox[2] < bbox[0]:
                bbox[0], bbox[2] = bbox[2], bbox[0]
            bbox = [float(b) for b in bbox]
            row = ColumnLine(polygon=bbox, vertical=vertical_line, horizontal=horizontal_line)
            line_info.append(row)

    if vertical:
        line_info = [line for line in line_info if line.vertical]

    if horizontal:
        line_info = [line for line in line_info if line.horizontal]

    return line_info


def get_vertical_lines(image, processor_size, image_size, divisor=20, x_tolerance=40, y_tolerance=20) -> List[ColumnLine]:
    vertical_lines = get_detected_lines(image, vertical=True)
    for line in vertical_lines:
        line.rescale(processor_size, image_size)
    vertical_lines = sorted(vertical_lines, key=lambda x: x.bbox[0])
    for line in vertical_lines:
        line.round(divisor)

    # Merge adjacent line segments together
    to_remove = []
    for i, line in enumerate(vertical_lines):
        for j, line2 in enumerate(vertical_lines):
            if j <= i:
                continue
            if line.bbox[0] != line2.bbox[0]:
                continue

            expanded_line1 = [line.bbox[0], line.bbox[1] - y_tolerance, line.bbox[2],
                              line.bbox[3] + y_tolerance]

            line1_points = set(range(int(expanded_line1[1]), int(expanded_line1[3])))
            line2_points = set(range(int(line2.bbox[1]), int(line2.bbox[3])))
            intersect_y = len(line1_points.intersection(line2_points)) > 0

            if intersect_y:
                vertical_lines[j].bbox[1] = min(line.bbox[1], line2.bbox[1])
                vertical_lines[j].bbox[3] = max(line.bbox[3], line2.bbox[3])
                to_remove.append(i)

    vertical_lines = [line for i, line in enumerate(vertical_lines) if i not in to_remove]

    # Remove redundant segments
    to_remove = []
    for i, line in enumerate(vertical_lines):
        if i in to_remove:
            continue
        for j, line2 in enumerate(vertical_lines):
            if j <= i or j in to_remove:
                continue
            close_in_x = abs(line.bbox[0] - line2.bbox[0]) < x_tolerance
            line1_points = set(range(int(line.bbox[1]), int(line.bbox[3])))
            line2_points = set(range(int(line2.bbox[1]), int(line2.bbox[3])))

            intersect_y = len(line1_points.intersection(line2_points)) > 0

            if close_in_x and intersect_y:
                # Keep the longer line and extend it
                if len(line2_points) > len(line1_points):
                    vertical_lines[j].bbox[1] = min(line.bbox[1], line2.bbox[1])
                    vertical_lines[j].bbox[3] = max(line.bbox[3], line2.bbox[3])
                    to_remove.append(i)
                else:
                    vertical_lines[i].bbox[1] = min(line.bbox[1], line2.bbox[1])
                    vertical_lines[i].bbox[3] = max(line.bbox[3], line2.bbox[3])
                    to_remove.append(j)

    vertical_lines = [line for i, line in enumerate(vertical_lines) if i not in to_remove]

    if len(vertical_lines) > 0:
        # Always start with top left of page
        vertical_lines[0].bbox[1] = 0

    return vertical_lines