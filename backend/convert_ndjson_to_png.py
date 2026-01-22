import json
from pathlib import Path
from PIL import Image, ImageDraw

# -----------------------------
# Pfade
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
NDJSON_DIR = BASE_DIR / "data" / "ndjson"
OUT_DIR = BASE_DIR / "data" / "images"

# -----------------------------
# Parameter
# -----------------------------
OUTPUT_SIZE = 64     
PADDING = 10          
LINE_WIDTH = 6       
MAX_IMAGES = 1500   

CANVAS_LIMIT = 255   


def draw_strokes(strokes):
    all_x, all_y = [], []

    for stroke in strokes:
        if len(stroke) < 2:
            continue
        xs = stroke[0]
        ys = stroke[1]

        all_x.extend([int(x) for x in xs])
        all_y.extend([int(y) for y in ys])

    if len(all_x) < 2 or len(all_y) < 2:
        return Image.new("L", (OUTPUT_SIZE, OUTPUT_SIZE), 255)

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    min_x = max(0, min_x - PADDING)
    min_y = max(0, min_y - PADDING)
    max_x = min(CANVAS_LIMIT, max_x + PADDING)
    max_y = min(CANVAS_LIMIT, max_y + PADDING)

    if max_x <= min_x or max_y <= min_y:
        return Image.new("L", (OUTPUT_SIZE, OUTPUT_SIZE), 255)

    width = int(max_x - min_x + 1)
    height = int(max_y - min_y + 1)

    img = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(img)

    for stroke in strokes:
        if len(stroke) < 2:
            continue
        xs = stroke[0]
        ys = stroke[1]

        points = [
            (int(x) - min_x, int(y) - min_y)
            for x, y in zip(xs, ys)
        ]

        if len(points) > 1:
            draw.line(points, fill=0, width=LINE_WIDTH)

    side = max(width, height)
    square = Image.new("L", (side, side), 255)
    square.paste(
        img,
        ((side - width) // 2, (side - height) // 2)
    )

    return square.resize(
        (OUTPUT_SIZE, OUTPUT_SIZE),
        Image.Resampling.BILINEAR
    )


# -----------------------------
# Datei verarbeiten
# -----------------------------
def convert_file(path: Path):
    class_name = path.stem.replace("full_raw_", "")
    out_dir = OUT_DIR / class_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Konvertiere {class_name} …")

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= MAX_IMAGES:
                break

            try:
                data = json.loads(line)
                img = draw_strokes(data["drawing"])
                img.save(out_dir / f"{class_name}_{i:05}.png")
            except Exception as e:
                continue


# -----------------------------
# Main
# -----------------------------
def main():
    files = list(NDJSON_DIR.glob("*.ndjson"))
    if not files:
        print("❌ Keine NDJSON-Dateien gefunden")
        return

    for f in files:
        convert_file(f)

    print("✅ Konvertierung abgeschlossen")


if __name__ == "__main__":
    main()
