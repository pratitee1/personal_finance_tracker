import cv2, pytesseract, numpy as np
from PIL import Image

LANGS = "deu+eng+chi_sim"
CFG_MAIN  = "--oem 3 --psm 6"
CFG_SMALL = "--oem 3 --psm 11"

def _remove_logo_band(img, band_px=70):
    return img[band_px:, :]

def _crop_receipt(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cnt  = max(cv2.findContours(th,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0],key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(cnt)
    return img[y:y+h, x:x+w]

def _clahe(gray):
    return cv2.createCLAHE(2.0,(8,8)).apply(gray)

def extract_text_blocks(path: str) -> list[str]:
    img   = cv2.imread(path)
    crop  = _crop_receipt(img)
    crop  = _remove_logo_band(crop)         
    gray  = _clahe(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
    h, w  = gray.shape
    if max(h, w) < 1800:
        s = 1800/max(h, w)
        gray = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    data = pytesseract.image_to_data(
        Image.fromarray(gray), lang=LANGS,
        config=CFG_MAIN, output_type=pytesseract.Output.DICT)
    roi   = gray[int(h*0.55):]             
    extra = pytesseract.image_to_data(
        Image.fromarray(roi), lang=LANGS,
        config=CFG_SMALL, output_type=pytesseract.Output.DICT)
    all_words  = data["text"]  + extra["text"]
    all_lines  = data["line_num"] + [ln+max(data["line_num"])+1
                                     for ln in extra["line_num"]]
    merged={}
    for word, ln in zip(all_words, all_lines):
        if word.strip():
            merged.setdefault(ln, []).append(word.strip())
    clean=[]
    for txt in merged.values():
        toks=[t for t in txt
              if (len(t) > 2 or any(c.isdigit() for c in t))
              and not (t.isupper() and len(t)<=4 and t.isalpha())]
        if toks:
            clean.append(" ".join(toks))
    return clean
