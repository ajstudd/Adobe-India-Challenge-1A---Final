import os
import sys
import csv
import unicodedata
# Ensure project root is in sys.path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.utils.pdf_utils import extract_pdf_text_with_fonts
from src.utils.text_utils import contains_mojibake, clean_text

INPUT_DIR = "local_dataset"
OUTPUT_DIR = "output/training_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_pdf(pdf_path, output_csv):
    import fitz
    import cv2
    import pytesseract
    from PIL import Image
    import numpy as np
    doc = fitz.open(pdf_path)
    blocks = []
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        page_height = page.rect.height
        # Gather all font sizes on this page for relative font size
        font_sizes = []
        page_dict = page.get_text("dict")
        for b in page_dict.get("blocks", []):
            if "lines" not in b:
                continue
            for line in b["lines"]:
                for span in line["spans"]:
                    if span["text"].strip():
                        font_sizes.append(span["size"])
        median_font_size = float(np.median(font_sizes)) if font_sizes else 1.0
        # For heading level: get unique font sizes descending
        unique_font_sizes = sorted(set(font_sizes), reverse=True)
        font_size_to_level = {}
        # Assign 1 to largest, 2 to second, 3 to third, 4 to fourth, 0 to others
        if unique_font_sizes:
            for idx, fs in enumerate(unique_font_sizes[:4]):
                font_size_to_level[fs] = idx + 1  # 1,2,3,4
        found_block = False
        prev_y1 = None
        prev_heading_idx = None
        for b in page_dict.get("blocks", []):
            if "lines" not in b:
                continue
            for line_idx, line in enumerate(b["lines"]):
                line_text = ""
                first_span = None
                bbox = [None, None, None, None]
                font = None
                color = None
                bold = italic = underline = False
                for span in line["spans"]:
                    text = span["text"].strip()
                    if not text:
                        continue
                    if first_span is None:
                        first_span = span
                        bbox = [span["bbox"][0], span["bbox"][1], span["bbox"][2], span["bbox"][3]]
                        font = span.get("font", None)
                        color = span.get("color", None)
                        bold = span.get("flags", 0) & 2 != 0
                        italic = span.get("flags", 0) & 1 != 0
                        underline = span.get("flags", 0) & 4 != 0
                    else:
                        bbox[0] = min(bbox[0], span["bbox"][0])
                        bbox[1] = min(bbox[1], span["bbox"][1])
                        bbox[2] = max(bbox[2], span["bbox"][2])
                        bbox[3] = max(bbox[3], span["bbox"][3])
                    line_text += (" " if line_text and not line_text[-1].isspace() and not text[0].isspace() else "") + text
                if line_text and first_span:
                    found_block = True
                    # Text features
                    is_all_caps = int(line_text.isupper())
                    is_title_case = int(line_text.istitle())
                    ends_with_colon = int(line_text.strip().endswith(":"))
                    starts_with_number = int(bool(line_text.strip() and line_text.strip().split()[0][0].isdigit()))
                    punctuation_count = sum(1 for c in line_text if c in '.,;:!?-—()[]{}"\'')
                    contains_colon = int(':' in line_text)
                    contains_semicolon = int(';' in line_text)
                    word_count = len(line_text.split())
                    # Visual features
                    y0 = bbox[1]
                    y1 = bbox[3]
                    line_position_on_page = y0 / page_height if page_height else 0
                    font_size = first_span["size"]
                    relative_font_size = font_size / median_font_size if median_font_size else 1.0
                    # Context features
                    distance_to_previous_heading = None
                    if prev_heading_idx is not None:
                        distance_to_previous_heading = line_idx - prev_heading_idx
                    line_spacing_above = None
                    if prev_y1 is not None:
                        line_spacing_above = y0 - prev_y1
                    # Heuristic heading level assignment (0=normal, 1=largest, 2=second, 3=third, 4=fourth)
                    heading_level = 0
                    if font_size in font_size_to_level:
                        # Only assign if text is short and not a bullet/math/hyperlink
                        if word_count <= 12 and not (line_text.strip().startswith(("•", "-", "‣", "*")) or any(sym in line_text for sym in ["=", "+", "-", "×", "÷", "∑", "∫", "√", "π", "∞"]) or line_text.strip().startswith("http") or "www." in line_text or ".com" in line_text):
                            heading_level = font_size_to_level[font_size]
                    # Always set heading_level to 0 if is_heading is 0 (after manual labeling)
                    blocks.append({
                        "text": line_text,
                        "font_size": font_size,
                        "page": page_num + 1,
                        "x0": bbox[0],
                        "y0": y0,
                        "x1": bbox[2],
                        "y1": y1,
                        "font": font,
                        "bold": bold,
                        "italic": italic,
                        "underline": underline,
                        "color": color,
                        "bullet": line_text.strip().startswith(("•", "-", "‣", "*")),
                        "math": any(sym in line_text for sym in ["=", "+", "-", "×", "÷", "∑", "∫", "√", "π", "∞"]),
                        "hyperlink": line_text.strip().startswith("http") or "www." in line_text or ".com" in line_text,
                        "is_all_caps": is_all_caps,
                        "is_title_case": is_title_case,
                        "ends_with_colon": ends_with_colon,
                        "starts_with_number": starts_with_number,
                        "punctuation_count": punctuation_count,
                        "contains_colon": contains_colon,
                        "contains_semicolon": contains_semicolon,
                        "word_count": word_count,
                        "line_position_on_page": line_position_on_page,
                        "relative_font_size": relative_font_size,
                        "distance_to_previous_heading": distance_to_previous_heading,
                        "line_spacing_above": line_spacing_above,
                        "is_heading": 0,  # Prefill with 0 for all rows
                        "heading_level": heading_level  # 0=normal, 1=largest, 2=second, 3=third, 4=fourth
                    })
                    prev_y1 = y1
        # OCR fallback (unchanged, but add new fields as None/False)
        if not found_block:
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            ocr_text = pytesseract.image_to_string(img_cv, lang="eng+hin+jpn")
            for line in ocr_text.splitlines():
                line_cleaned = line.strip()
                if line_cleaned:
                    blocks.append({
                        "text": line_cleaned,
                        "font_size": None,
                        "page": page_num + 1,
                        "x0": None,
                        "y0": None,
                        "x1": None,
                        "y1": None,
                        "font": None,
                        "bold": False,
                        "italic": False,
                        "underline": False,
                        "color": None,
                        "bullet": False,
                        "math": False,
                        "hyperlink": False,
                        "is_all_caps": None,
                        "is_title_case": None,
                        "ends_with_colon": None,
                        "starts_with_number": None,
                        "punctuation_count": None,
                        "line_position_on_page": None,
                        "relative_font_size": None,
                        "distance_to_previous_heading": None,
                        "line_spacing_above": None,
                        "is_heading": 0,
                        "heading_level": 0
                    })
    # Write to CSV for manual labelling, using utf-8-sig for Excel/multilingual compatibility
    with open(output_csv, mode="w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "text", "font_size", "page", "x0", "y0", "x1", "y1",
                "font", "bold", "italic", "underline", "color", "bullet", "math", "hyperlink",
                "is_all_caps", "is_title_case", "ends_with_colon", "starts_with_number", "punctuation_count",
                "contains_colon", "contains_semicolon", "word_count",
                "line_position_on_page", "relative_font_size", "distance_to_previous_heading", "line_spacing_above",
                "is_heading", "heading_level"
            ],
            quoting=csv.QUOTE_ALL,
            escapechar="\\"
        )
        writer.writeheader()
        for block in blocks:
            # Ensure is_heading is always prefilled as 0
            block['is_heading'] = 0
            writer.writerow(block)

def main():
    pdf_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".pdf")]
    for pdf_file in pdf_files:
        print(f"Processing {pdf_file}")
        pdf_path = os.path.join(INPUT_DIR, pdf_file)
        output_csv = os.path.join(OUTPUT_DIR, f"{pdf_file.replace('.pdf', '')}_blocks.csv")
        process_pdf(pdf_path, output_csv)
        print(f"Saved CSV for manual labelling: {output_csv}")

if __name__ == "__main__":
    main()
