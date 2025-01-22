from PIL import Image, ImageDraw

def test_table_rec(table_rec_predictor):
    data = [
        ["Name", "Age", "City"],
        ["Alice", 25, "New York"],
        ["Bob", 30, "Los Angeles"],
        ["Charlie", 35, "Chicago"],
    ]
    test_image = draw_table(data)

    results = table_rec_predictor([test_image])
    assert len(results) == 1
    assert results[0].image_bbox == [0, 0, test_image.size[0], test_image.size[1]]

    cells = results[0].cells
    assert len(cells) == 12
    for row_id in range(4):
        for col_id in range(3):
            cell = [c for c in cells if c.row_id == row_id and c.col_id == col_id]
            assert len(cell) == 1, f"Missing cell at row {row_id}, col {col_id}"

def draw_table(data, cell_width=100, cell_height=40):
    rows = len(data)
    cols = len(data[0])
    width = cols * cell_width
    height = rows * cell_height

    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)

    for i in range(rows + 1):
        y = i * cell_height
        draw.line([(0, y), (width, y)], fill='black', width=1)

    for i in range(cols + 1):
        x = i * cell_width
        draw.line([(x, 0), (x, height)], fill='black', width=1)

    for i in range(rows):
        for j in range(cols):
            text = str(data[i][j])
            text_bbox = draw.textbbox((0, 0), text)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            x = j * cell_width + (cell_width - text_width) // 2
            y = i * cell_height + (cell_height - text_height) // 2

            draw.text((x, y), text, fill='black')

    return image