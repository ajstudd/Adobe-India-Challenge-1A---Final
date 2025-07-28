import os
import json
import fitz
import csv
from jsonschema import validate
from jsonschema.exceptions import ValidationError
import joblib
from src.utils.pdf_utils import extract_pdf_text_with_fonts
from src.utils.text_utils import is_valid_heading, clean_text
USE_ML = os.getenv("USE_ML", "false").lower() == "true"
if USE_ML:
    model = joblib.load("models/heading_classifier.pkl")

# For Docker
# OUTPUT_DIR = "/app/output"
# INPUT_DIR = "/app/input"

# Local Development
OUTPUT_DIR = "output"
INPUT_DIR = "input"

SCHEMA_PATH = "src/schema/output_schema.json"

def run_outline_extraction():
    """
    Process all PDFs in the input directory and extract outlines.
    """
    pdf_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".pdf")]

    for pdf_file in pdf_files:
        print(f"[INFO] Processing {pdf_file}")
        
        pdf_path = os.path.join(INPUT_DIR, pdf_file)
        outline = extract_outline_from_pdf(pdf_path)
        
        try:
            validate_output_schema(outline, SCHEMA_PATH)
            print(f"[INFO] Output for {pdf_file} is valid ✅")
        except ValidationError as e:
            print(f"[ERROR] Schema validation failed for {pdf_file}")
            print(e)
            continue
        
        # Write output JSON
        output_filename = pdf_file.replace(".pdf", ".json")
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        with open(output_path, "w", encoding="utf-8") as out_file:
            json.dump(outline, out_file, indent=2, ensure_ascii=False)

        print(f"[INFO] Saved output to {output_filename}")


def export_blocks_to_csv(blocks, pdf_filename):
    training_dir = os.path.join(OUTPUT_DIR, "training_data")
    os.makedirs(training_dir, exist_ok=True)

    csv_file = os.path.join(training_dir, f"{pdf_filename.replace('.pdf', '')}_blocks.csv")

    with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "text", "font_size", "page", "x0", "y0", "x1", "y1",
                "font", "bold", "italic", "underline", "color", "bullet", "math", "hyperlink", "is_heading"
            ],
            quoting=csv.QUOTE_ALL,
            escapechar="\\"
        )
        writer.writeheader()
        for block in blocks:
            writer.writerow({
                "text": block["text"],
                "font_size": block["font_size"],
                "page": block["page"],
                "x0": block.get("x0", 0),
                "y0": block.get("y0", 0),
                "x1": block.get("x1", 0),
                "y1": block.get("y1", 0),
                "font": block.get("font", ""),
                "bold": block.get("bold", False),
                "italic": block.get("italic", False),
                "underline": block.get("underline", False),
                "color": block.get("color", None),
                "bullet": block.get("bullet", False),
                "math": block.get("math", False),
                "hyperlink": block.get("hyperlink", False),
                "is_heading": block.get("is_heading", "")  # Manual label, to be filled in later
            })

def extract_outline_from_pdf(pdf_path):
    """
    Extract hierarchical headings from a single PDF file.
    """
    doc = fitz.open(pdf_path)
    document_title = doc.metadata.get("title", os.path.basename(pdf_path).replace(".pdf", ""))
    
    # Collect text blocks across all pages
    all_blocks = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = extract_pdf_text_with_fonts(page)
        for block in blocks:
            block['page'] = page_num
            all_blocks.append(block)
    
    # Sort and classify headings
    export_blocks_to_csv(all_blocks, os.path.basename(pdf_path))
    headings = classify_headings(all_blocks)

    return {
        "title": document_title,
        "outline": headings
    }

def is_heading_ml(text, font_size):
    if not USE_ML:
        # Fallback to simple rule-based check
        return len(text) < 100 and not text.endswith(".")
    
    text_len = len(text)
    num_words = len(text.split())
    return model.predict([(text, [[font_size, text_len, num_words]])])[0] == 1

def classify_headings(blocks):
    """
    Use font size and semantics to assign H1, H2, H3 levels.
    """
    # Sort by font size to detect hierarchy
    sizes = sorted(list(set([b['font_size'] for b in blocks])), reverse=True)
    
    if len(sizes) < 3:
        # Fall back if not enough font variation
        levels = {s: f"H{idx+1}" for idx, s in enumerate(sizes)}
    else:
        levels = {sizes[0]: "H1", sizes[1]: "H2", sizes[2]: "H3"}

    outline = []

    for block in blocks:
        text = clean_text(block['text'])
        if not is_heading_ml(text, block['font_size']):
            continue
        
        font_size = block['font_size']
        page = block['page']
        
        level = levels.get(font_size, None)
        if level is None:
            continue

        outline.append({
            "level": level,
            "text": text,
            "page": page
        })

    return outline

def validate_output_schema(output_data, schema_path):
    """
    Validates output JSON against the provided schema.
    """
    with open(schema_path, "r") as schema_file:
        schema = json.load(schema_file)

    validate(instance=output_data, schema=schema)
