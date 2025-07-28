def auto_label_heading(block):
    """
    Heuristic: assign is_heading=1 if block likely a heading, else 0.
    Criteria: large font, bold, short text, title case, not bullet/math/hyperlink.
    """
    # Improved logic: use more features and thresholds
    if block["font_size"] >= 14 and block["text_length"] <= 80:
        if block["bold"] or block["is_title_case"] or block["is_uppercase"]:
            if not block["bullet"] and not block["math"] and not block["hyperlink"]:
                if not block["contains_digits"] and not block["contains_punctuation"]:
                    return 1
    # Also consider headings with large font and short text, even if not bold
    if block["font_size"] >= 16 and block["text_length"] <= 60 and block["num_words"] <= 8:
        if not block["bullet"] and not block["math"] and not block["hyperlink"]:
            return 1
    return 0
from src.utils.text_utils import contains_mojibake, clean_text
import unicodedata

def extract_pdf_text_with_fonts(page):
    """
    Extracts text blocks along with font size from PyMuPDF page.
    Attempts to recover encoding issues safely.
    """
    blocks = []

    # No font skipping: extract all blocks for maximal training data

    for b in page.get_text("dict")["blocks"]:
        if "lines" not in b:
            continue
        for line in b["lines"]:
            # Merge all spans in a line into a single block
            line_text = ""
            first_span = None
            bbox = [None, None, None, None]
            for span in line["spans"]:
                text = span["text"]
                text = unicodedata.normalize("NFKC", text)
                text = clean_text(text)
                if not text:
                    continue
                if first_span is None:
                    first_span = span
                    bbox = [span["bbox"][0], span["bbox"][1], span["bbox"][2], span["bbox"][3]]
                else:
                    bbox[0] = min(bbox[0], span["bbox"][0])
                    bbox[1] = min(bbox[1], span["bbox"][1])
                    bbox[2] = max(bbox[2], span["bbox"][2])
                    bbox[3] = max(bbox[3], span["bbox"][3])
                line_text += (" " if line_text and not line_text[-1].isspace() and not text[0].isspace() else "") + text
            if line_text and first_span:
                # Feature extraction
                text_length = len(line_text)
                num_words = len(line_text.split())
                is_title = line_text.istitle()
                is_upper = line_text.isupper()
                contains_digits = any(c.isdigit() for c in line_text)
                contains_punct = any(c in "!@#$%^&*()[]{};:,./<>?\\|`~-=+" for c in line_text)
                # Simple language detection: check for non-ASCII
                non_ascii = sum(1 for c in line_text if ord(c) > 127)
                language = "non-english" if non_ascii > 0 else "english"
                block = {
                    "text": line_text,
                    "font_size": first_span["size"],
                    "font": first_span.get("font", ""),
                    "bold": first_span.get("flags", 0) & 2 != 0,
                    "italic": first_span.get("flags", 0) & 1 != 0,
                    "underline": first_span.get("flags", 0) & 4 != 0,
                    "color": first_span.get("color", None),
                    "bullet": line_text.strip().startswith(("•", "-", "‣", "*")),
                    "math": any(sym in line_text for sym in ["=", "+", "-", "×", "÷", "∑", "∫", "√", "π", "∞"]),
                    "hyperlink": line_text.strip().startswith("http") or "www." in line_text or ".com" in line_text,
                    "x0": bbox[0],
                    "y0": bbox[1],
                    "x1": bbox[2],
                    "y1": bbox[3],
                    "text_length": text_length,
                    "num_words": num_words,
                    "is_title_case": is_title,
                    "is_uppercase": is_upper,
                    "contains_digits": contains_digits,
                    "contains_punctuation": contains_punct,
                    "language": language,
                    "page": page.number + 1 if hasattr(page, 'number') else None
                }
                block["is_heading"] = auto_label_heading(block)
                blocks.append(block)

    return blocks
