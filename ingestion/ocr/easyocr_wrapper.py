import easyocr
import numpy as np
import cv2
from typing import List, Tuple

READER = easyocr.Reader(
    ["de", "en"], gpu=False, verbose=False
)

def sort_by_reading_order(results) -> List[Tuple[str, List[int]]]:
    sorted_res = sorted(
        results,
        key=lambda r: (
            min(p[1] for p in r[0]),   # top-most y
            min(p[0] for p in r[0]),   # left-most x
        ),
    )
    lines = []
    current_y = -1
    buf = []
    for bbox, text, conf in sorted_res:
        if conf < 0.3 or not text.strip():
            continue
        y_top = min(p[1] for p in bbox)
        if current_y != -1 and y_top - current_y > 15:  
            lines.append(" ".join(buf))
            buf = []
        buf.append(text.strip())
        current_y = y_top
    if buf:
        lines.append(" ".join(buf))
    return lines

def extract_lines(img_path: str) -> List[str]:
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    results = READER.readtext(gray, detail=1, paragraph=False)
    return sort_by_reading_order(results)

