from typing import List

from pydantic import BaseModel

from surya.common.polygon import PolygonBox


class TableCell(PolygonBox):
    row_id: int
    colspan: int
    within_row_id: int
    cell_id: int
    is_header: bool
    rowspan: int | None = None
    merge_up: bool = False
    merge_down: bool = False
    col_id: int | None = None
    text_lines: List[dict] | None = None

    @property
    def label(self):
        return f'Cell {self.cell_id} {self.rowspan}/{self.colspan}'


class TableRow(PolygonBox):
    row_id: int
    is_header: bool

    @property
    def label(self):
        return f'Row {self.row_id}'


class TableCol(PolygonBox):
    col_id: int
    is_header: bool

    @property
    def label(self):
        return f'Column {self.col_id}'


class TableResult(BaseModel):
    cells: List[TableCell]
    unmerged_cells: List[TableCell]
    rows: List[TableRow]
    cols: List[TableCol]
    image_bbox: List[float]
