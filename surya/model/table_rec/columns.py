from collections import defaultdict
from statistics import mean
from surya.schema import TableCol


def find_columns(rows, cells):
    column_cells = defaultdict(list)
    for row in rows:
        row_cells = [c for c in cells if c.row_id == row.row_id]
        row_cells = sorted(row_cells, key=lambda x: x.within_row_id)
        colspan = 0
        for cell in row_cells:
            column_cells[cell.within_row_id + colspan].append(cell)
            colspan += cell.colspan - 1

    columns = []
    for i, idx in enumerate(column_cells):
        bbox = [
            min(c.bbox[0] for c in column_cells[idx]),
            min(c.bbox[1] for c in column_cells[idx]),
            max(c.bbox[2] for c in column_cells[idx]),
            max(c.bbox[3] for c in column_cells[idx]),
        ]

        col = TableCol(
            polygon=[[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]]],
            col_id=i,
        )
        columns.append(col)
    return columns





