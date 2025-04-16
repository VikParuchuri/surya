from typing import List

import torch


def prediction_to_polygon_batch(
    pred: List[torch.tensor], img_size, bbox_scaler, skew_scaler, skew_min=0.001
):
    pred = torch.stack(pred, dim=0)
    w_scale = img_size[0] / bbox_scaler
    h_scale = img_size[1] / bbox_scaler

    cx = pred[:, 0]
    cy = pred[:, 1]
    width = pred[:, 2]
    height = pred[:, 3]

    x1 = cx - width / 2
    y1 = cy - height / 2
    x2 = cx + width / 2
    y2 = cy + height / 2

    skew_x = torch.floor((pred[:, 4] - skew_scaler) / 2)
    skew_y = torch.floor((pred[:, 5] - skew_scaler) / 2)

    skew_x[torch.abs(skew_x) < skew_min] = 0
    skew_y[torch.abs(skew_y) < skew_min] = 0

    polygons_flat = torch.stack(
        [
            x1 - skew_x,
            y1 - skew_y,
            x2 - skew_x,
            y1 + skew_y,
            x2 + skew_x,
            y2 + skew_y,
            x1 + skew_x,
            y2 - skew_y,
        ],
        dim=1,
    )

    batch_size = pred.size(0)
    polygons = polygons_flat.view(batch_size, 4, 2)

    polygons[:, :, 0] *= w_scale
    polygons[:, :, 1] *= h_scale

    return polygons.tolist()


def prediction_to_polygon(pred, img_size, bbox_scaler, skew_scaler, skew_min=0.001):
    w_scale = img_size[0] / bbox_scaler
    h_scale = img_size[1] / bbox_scaler

    boxes = pred
    cx = boxes[0]
    cy = boxes[1]
    width = boxes[2]
    height = boxes[3]
    x1 = cx - width / 2
    y1 = cy - height / 2
    x2 = cx + width / 2
    y2 = cy + height / 2
    skew_x = torch.floor((boxes[4] - skew_scaler) / 2)
    skew_y = torch.floor((boxes[5] - skew_scaler) / 2)

    # Ensures we don't get slightly warped boxes
    # Note that the values are later scaled, so this is in 1/1024 space
    skew_x[torch.abs(skew_x) < skew_min] = 0
    skew_y[torch.abs(skew_y) < skew_min] = 0

    polygon = [
        x1 - skew_x,
        y1 - skew_y,
        x2 - skew_x,
        y1 + skew_y,
        x2 + skew_x,
        y2 + skew_y,
        x1 + skew_x,
        y2 - skew_y,
    ]
    poly = []
    for i in range(4):
        poly.append(
            [polygon[2 * i].item() * w_scale, polygon[2 * i + 1].item() * h_scale]
        )
    return poly
