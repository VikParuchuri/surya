import math
from typing import List, Tuple
from PIL import Image

from surya.layout.schema import LayoutResult

SLICES_TYPE = Tuple[List[Image.Image], List[Tuple[int, int, int]]]


class ImageSlicer:
    merge_tolerance = .05
    merge_margin = .05

    def __init__(self, slice_min_dims, slice_sizes, max_slices=4):
        self.slice_min_dims = slice_min_dims
        self.max_slices = max_slices
        self.slice_sizes = slice_sizes

    def slice(self, images: List[Image.Image]) -> SLICES_TYPE:
        all_slices = []
        all_positions = []

        for idx, image in enumerate(images):
            if (image.size[0] > self.slice_min_dims["width"] or
                    image.size[1] > self.slice_min_dims["height"]):
                img_slices, positions = self._slice_image(image, idx)
                all_slices.extend(img_slices)
                all_positions.extend(positions)
            else:
                all_slices.append(image)
                all_positions.append((idx, 0, 0))

        return all_slices, all_positions

    def slice_count(self, image: Image.Image) -> int:
        width, height = image.size
        if width > height:
            slice_size = self._calculate_slice_size(width, "width")
            return math.ceil(width / slice_size)
        else:
            slice_size = self._calculate_slice_size(height, "height")
            return math.ceil(height / slice_size)

    def _calculate_slice_size(self, dimension: int, dim_type: str) -> int:
        min_size = self.slice_sizes[dim_type]
        return max(min_size, (dimension // self.max_slices + 1))

    def _slice_image(self, image: Image.Image, idx: int) -> SLICES_TYPE:
        width, height = image.size
        slices = []
        positions = []

        if width > height:
            slice_size = self._calculate_slice_size(width, "width")
            for i, x in enumerate(range(0, width, slice_size)):
                slice_end = min(x + slice_size, width)
                slices.append(image.crop((x, 0, slice_end, height)))
                positions.append((idx, i, 0))
        else:
            slice_size = self._calculate_slice_size(height, "height")
            for i, y in enumerate(range(0, height, slice_size)):
                slice_end = min(y + slice_size, height)
                slices.append(image.crop((0, y, width, slice_end)))
                positions.append((idx, 0, i))

        return slices, positions

    def join(self, results: List[LayoutResult], tile_positions: List[Tuple[int, int, int]]) -> List[LayoutResult]:
        new_results = []
        current_result = None
        for idx, (result, tile_position) in enumerate(zip(results, tile_positions)):
            image_idx, tile_x, tile_y = tile_position
            if idx == 0 or image_idx != tile_positions[idx - 1][0]:
                if current_result is not None:
                    new_results.append(current_result)
                current_result = result
            else:
                merge_dir = "width" if tile_x > 0 else "height"
                current_result = self.merge_results(current_result, result, merge_dir=merge_dir)
        if current_result is not None:
            new_results.append(current_result)
        return new_results

    def merge_results(self, res1: LayoutResult, res2: LayoutResult, merge_dir="width") -> LayoutResult:
        new_image_bbox = res1.image_bbox.copy()
        to_remove_idxs = set()
        if merge_dir == "width":
            new_image_bbox[2] += res2.image_bbox[2]
            max_position = max([box.position for box in res1.bboxes]) + 1
            for i, box2 in enumerate(res2.bboxes):
                box2.shift(x_shift=res1.image_bbox[2])
                box2.position += max_position
                for j, box1 in enumerate(res1.bboxes):
                    if all([
                        any([
                            box1.intersection_pct(box2, x_margin=self.merge_margin) > self.merge_tolerance,
                            box2.intersection_pct(box1, x_margin=self.merge_margin) > self.merge_tolerance
                        ]),
                        any([
                            box1.y_overlap(box2) > box1.height // 2,
                            box2.y_overlap(box1) > box2.height // 2
                        ]),
                        any([
                            box1.label == box2.label,
                            (box1.label in ["Picture", "Figure"] and box2.label in ["Picture", "Figure"])
                        ])
                    ]):
                        box1.merge(box2)
                        to_remove_idxs.add(i)

        elif merge_dir == "height":
            new_image_bbox[3] += res2.image_bbox[3]
            max_position = max([box.position for box in res1.bboxes]) + 1
            for i, box2 in enumerate(res2.bboxes):
                box2.shift(y_shift=res1.image_bbox[3])
                box2.position += max_position
                for j, box1 in enumerate(res1.bboxes):
                    if all([
                        any([
                            box1.intersection_pct(box2, y_margin=self.merge_margin) > self.merge_tolerance,
                            box2.intersection_pct(box1, y_margin=self.merge_margin) > self.merge_tolerance
                        ]),
                        any([
                            box1.x_overlap(box2) > box1.width // 2,
                            box2.x_overlap(box1) > box2.width // 2
                        ]),
                        any([
                            box1.label == box2.label,
                            (box1.label in ["Picture", "Figure"] and box2.label in ["Picture", "Figure"])
                        ])
                    ]):
                        box1.merge(box2)
                        to_remove_idxs.add(i)

        new_result = LayoutResult(
            image_bbox=new_image_bbox,
            bboxes=res1.bboxes + [b for i, b in enumerate(res2.bboxes) if i not in to_remove_idxs],
            sliced=True,
        )
        return new_result