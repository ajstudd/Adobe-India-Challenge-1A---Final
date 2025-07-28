#!/usr/bin/env python3
"""
Direct hierarchical numbering JSON generation
bypassing incorrect ML model predictions
Now processes all PDFs in input folder and generates JSON files following output_schema.json
"""

import pandas as pd
import json
import sys
import os
import re
import glob
import fitz  # PyMuPDF
import numpy as np
from pathlib import Path

# Optional OCR imports
try:
    import cv2
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

sys.path.append('.')

def extract_pdf_blocks(pdf_path):
    """Extract blocks from PDF using PyMuPDF with OCR fallback for garbage text"""
    print(f"📄 Extracting blocks from {os.path.basename(pdf_path)}...")
    
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
        page_blocks = []  # Store blocks for this page
        
        # First pass: extract using programmatic method
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
                    
                    # Clean up text - remove control characters but preserve basic punctuation
                    text = re.sub(r'[\u0000-\u0008\u000B-\u001F\u007F-\u009F]', '', text)
                    text = text.strip()
                    
                    if not text:  # Skip if text becomes empty after cleaning
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
                    
                    # Heuristic heading level assignment
                    heading_level = 0
                    if font_size in font_size_to_level:
                        # Only assign if text is short and not a bullet/math/hyperlink
                        if word_count <= 12 and not (line_text.strip().startswith(("•", "-", "‣", "*")) or 
                                                    any(sym in line_text for sym in ["=", "+", "-", "×", "÷", "∑", "∫", "√", "π", "∞"]) or 
                                                    line_text.strip().startswith("http") or "www." in line_text or ".com" in line_text):
                            heading_level = font_size_to_level[font_size]
                    
                    page_blocks.append({
                        "text": line_text,
                        "font_size": font_size,
                        "page": page_num + 1,  # 1-based page numbering
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
                        "heading_level": heading_level,  # 0=normal, 1=largest, 2=second, 3=third, 4=fourth
                        "extraction_method": "programmatic"
                    })
                    prev_y1 = y1
        
        # Check if this page has too much garbage text - if so, use OCR fallback
        garbage_count = 0
        total_text_blocks = len(page_blocks)
        
        for block in page_blocks:
            if not is_valid_heading_text(block["text"]) and len(block["text"]) > 5:
                garbage_count += 1
        
        garbage_ratio = garbage_count / total_text_blocks if total_text_blocks > 0 else 0
        
        # If more than 40% of text blocks are garbage, use OCR for this page
        if garbage_ratio > 0.4 and OCR_AVAILABLE:
            print(f"⚠️  Page {page_num + 1}: {garbage_ratio:.1%} garbage text detected, switching to OCR...")
            ocr_blocks = extract_page_with_ocr(page, page_num + 1, page_height, font_size_to_level, median_font_size)
            
            # Use OCR blocks instead of programmatic extraction for this page
            if ocr_blocks:
                blocks.extend(ocr_blocks)
                print(f"✅ Page {page_num + 1}: Used OCR, extracted {len(ocr_blocks)} blocks")
            else:
                # Fallback to programmatic extraction even if it's garbage
                blocks.extend(page_blocks)
                print(f"⚠️  Page {page_num + 1}: OCR failed, using programmatic extraction")
        else:
            # Use programmatic extraction
            blocks.extend(page_blocks)
            if garbage_ratio > 0:
                print(f"📝 Page {page_num + 1}: {garbage_ratio:.1%} garbage text, but below threshold")
    
    doc.close()
    
    if blocks:
        df = pd.DataFrame(blocks)
        print(f"✅ Extracted {len(df)} blocks from PDF")
        return df
    else:
        print("❌ No blocks extracted from PDF")
        return None

def extract_page_with_ocr(page, page_num, page_height, font_size_to_level, median_font_size):
    """Extract text from a page using OCR as fallback"""
    if not OCR_AVAILABLE:
        return []
    
    try:
        import cv2
        import pytesseract
        from PIL import Image
        
        # Convert page to image
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x scale for better OCR
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # Extract text with bounding boxes using OCR
        ocr_data = pytesseract.image_to_data(img_cv, output_type=pytesseract.Output.DICT, lang="eng")
        
        blocks = []
        current_line = ""
        current_bbox = None
        current_conf = 0
        
        for i in range(len(ocr_data['text'])):
            text = ocr_data['text'][i].strip()
            conf = int(ocr_data['conf'][i])
            
            if text and conf > 30:  # Only use text with reasonable confidence
                # Clean the OCR text
                text = re.sub(r'[\u0000-\u0008\u000B-\u001F\u007F-\u009F]', '', text)
                text = text.strip()
                
                if text and is_valid_heading_text(text):
                    x, y, w, h = ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i]
                    
                    # Scale coordinates back (we used 2x scaling)
                    x, y, w, h = x/2, y/2, w/2, h/2
                    
                    # Estimate font size from height
                    font_size = max(8, h * 0.75)  # Rough estimation
                    
                    # Determine heading level
                    heading_level = 0
                    if font_size > 20:
                        heading_level = 1
                    elif font_size > 16:
                        heading_level = 2
                    elif font_size > 14:
                        heading_level = 3
                    elif font_size > 12:
                        heading_level = 4
                    
                    blocks.append({
                        "text": text,
                        "font_size": font_size,
                        "page": page_num,
                        "x0": x,
                        "y0": y,
                        "x1": x + w,
                        "y1": y + h,
                        "font": None,
                        "bold": False,
                        "italic": False,
                        "underline": False,
                        "color": None,
                        "bullet": text.startswith(("•", "-", "‣", "*")),
                        "math": any(sym in text for sym in ["=", "+", "-", "×", "÷", "∑", "∫", "√", "π", "∞"]),
                        "hyperlink": text.startswith("http") or "www." in text or ".com" in text,
                        "is_all_caps": int(text.isupper()),
                        "is_title_case": int(text.istitle()),
                        "ends_with_colon": int(text.endswith(":")),
                        "starts_with_number": int(bool(text and text.split()[0][0].isdigit() if text.split() else False)),
                        "punctuation_count": sum(1 for c in text if c in '.,;:!?-—()[]{}"\''),
                        "contains_colon": int(':' in text),
                        "contains_semicolon": int(';' in text),
                        "word_count": len(text.split()),
                        "line_position_on_page": y / page_height if page_height else 0,
                        "relative_font_size": font_size / median_font_size if median_font_size else 1.0,
                        "distance_to_previous_heading": None,
                        "line_spacing_above": None,
                        "is_heading": 0,
                        "heading_level": heading_level,
                        "extraction_method": "ocr",
                        "ocr_confidence": conf
                    })
        
        return blocks
        
    except Exception as e:
        print(f"❌ OCR failed for page {page_num}: {e}")
        return []

def is_valid_heading_text(text):
    """Check if text is valid for heading detection (filter out garbage/corrupted text)"""
    
    # Basic length check
    if len(text.strip()) < 2:
        return False
    
    # Remove common unicode control characters and whitespace for analysis
    clean_text = re.sub(r'[\u0000-\u001F\u007F-\u009F]', '', text).strip()
    
    # If after removing control characters, text becomes too short or empty
    if len(clean_text) < 2:
        return False
    
    # Count control characters and special unicode characters
    control_char_count = len(text) - len(clean_text)
    control_char_ratio = control_char_count / len(text) if len(text) > 0 else 0
    
    # Reject if more than 30% of the text is control characters
    if control_char_ratio > 0.3:
        return False
    
    # Check for mostly numbers and special characters (like "5/\u0001 \u0001")
    alphanumeric_count = sum(1 for c in clean_text if c.isalnum())
    alphanumeric_ratio = alphanumeric_count / len(clean_text) if len(clean_text) > 0 else 0
    
    # Reject if less than 30% alphanumeric characters
    if alphanumeric_ratio < 0.3:
        return False
    
    # Reject pure numbers with minimal other content
    if re.match(r'^[\d\s/\-.,\u0001]+$', text):
        return False
    
    # Reject text that's mostly special characters or symbols
    special_char_patterns = [
        r'^[\d\s\u0001/\-.,;:!?()]+$',  # Mostly numbers, spaces, and basic punctuation
        r'^[^\w\s]*$',  # Only non-word characters
        r'^\s*[\u0001-\u001F]+\s*$',  # Only control characters and whitespace
    ]
    
    for pattern in special_char_patterns:
        if re.match(pattern, text):
            return False
    
    # Check for minimum word content
    words = clean_text.split()
    if len(words) == 0:
        return False
    
    # Check for excessive special characters
    special_chars = re.findall(r'[#@$%^&*!?()]', text)
    if len(special_chars) > len(text) * 0.4:  # More than 40% special chars
        return False
    
    # Check for OCR corruption indicators
    corruption_indicators = [
        '%' in text and len(text) > 20,  # Percent signs in long text (often OCR corruption)
        '?' in text and text.count('?') > 1,  # Multiple question marks
        ')/' in text,  # Patterns like ")/-"
        '/--' in text,  # Patterns like "/--6"
        'CXag' in text,  # Specific OCR corruption
        'WXW' in text,   # More specific OCR corruption
        'TfT' in text,   # More specific OCR corruption  
        'YXe' in text,   # More specific OCR corruption
        'bY' in text and len(text) > 15,  # Common OCR misread in long text
        'XaW' in text,   # More OCR corruption
        'Wg' in text and len(text) > 10,  # OCR corruption pattern
        re.search(r'[A-Z][a-z]{1,2}[A-Z][a-z]{1,2}[A-Z]', text),  # Mixed case OCR pattern
        re.search(r'\d[A-Z][a-z]{2,}[A-Z]', text),  # Number + mixed case
    ]
    
    if any(corruption_indicators):
        return False
    
    # For very short text, require at least one real word (not just numbers/symbols)
    if len(words) <= 2:
        has_real_word = any(
            len(word) >= 2 and re.search(r'[a-zA-Z]', word) 
            for word in words
        )
        if not has_real_word:
            return False
    
    # Additional OCR corruption patterns
    ocr_corruption_patterns = [
        r'^[A-Z]{1,2}\d+\s*$',  # Single letters followed by numbers
        r'^\d+[A-Z]{1,2}\s*$',  # Numbers followed by single letters
        r'^[^\w\s]{3,}$',       # Three or more consecutive non-word characters
        r'^\s*[|\\\/\-_=+<>~`]{2,}\s*$',  # Lines made of symbols
        r'^[A-Z0-9#!@$%^&*]{10,}$',  # Long strings of caps, numbers and symbols (OCR garbage)
        r'^[A-Z]{2,}\d{2,}[A-Z]{2,}',  # Mixed caps and numbers pattern
        r'[#!@$%^&*]{3,}',  # Multiple special characters in a row
        r'^[A-Z][a-z]*[A-Z][a-z]*[A-Z][a-z]*[A-Z]',  # Weird mixed case patterns
        r'^[FN\[\]Q]{4,}',  # Common OCR misreads
        
        # Enhanced patterns for the specific corruption you're seeing
        r'^[A-Z][a-z]*[A-Z][a-z]*%[A-Z]',  # Patterns like "9ZeXX CXag%TfT"
        r'[A-Z][a-z]{2,}[A-Z][a-z]{2,}[A-Z]',  # Mixed case like "CXag" "WXWXk"
        r'[A-Z]{2,}[a-z]{2,}[A-Z]{2,}',  # Another mixed case pattern
        r'^[A-Z\d][a-z]{2,}[A-Z][a-z]{1,3}[A-Z][a-z]{1,3}',  # Patterns like "9ZeXX"
        r'[?][a-z]{1,3}[A-Z][a-z]{1,3}',  # Question marks with mixed case
        r'%[A-Z][a-z]*[A-Z]',  # Percent signs with mixed case
        r'[)]/[-]{2,}[A-Z]',  # Patterns like ")/-/1)"
        r'[A-Z][a-z]{1,2}[A-Z][a-z]{1,2}[A-Z][a-z]{1,2}[A-Z]',  # Long mixed case sequences
        r'^[A-Z\d][a-z]{3,}[A-Z][a-z]{3,}[A-Z]',  # Specific OCR corruption pattern
        r'[A-Z][a-z]*![a-z]*[A-Z]',  # Exclamation marks in weird places
        r'[A-Z]{1,2}[a-z]{1,3}[A-Z]{1,2}[a-z]{1,3}[?!%]',  # Special chars at end of mixed case
        r'^[0-9][A-Z][a-z]{2,}[A-Z].*[A-Z][a-z]{2,}[0-9)]',  # Number + mixed case + number/paren
    ]
    
    for pattern in ocr_corruption_patterns:
        if re.search(pattern, text):  # Changed from match to search to catch patterns anywhere
            return False
    
    return True

def process_single_pdf(pdf_path, output_dir):
    """Process a single PDF and generate its JSON output"""
    pdf_name = Path(pdf_path).stem
    print(f"\n� Processing: {pdf_name}")
    print("=" * 50)
    
    # Extract blocks from PDF
    df = extract_pdf_blocks(pdf_path)
    if df is None or len(df) == 0:
        print(f"❌ Failed to extract blocks from {pdf_name}")
        return False
    
    # Import hierarchical numbering analyzer with fallback
    try:
        from hierarchical_numbering_analyzer import HierarchicalNumberingAnalyzer
        analyzer = HierarchicalNumberingAnalyzer()
        has_analyzer = True
    except ImportError as e:
        print(f"⚠️  Hierarchical numbering analyzer not available: {e}")
        analyzer = None
        has_analyzer = False
    
    try:
        from intelligent_filter import IntelligentFilter
        filter_system = IntelligentFilter()
        has_filter = True
    except ImportError as e:
        print(f"⚠️  Intelligent filter not available: {e}")
        filter_system = None
        has_filter = False
    
    # Identify headings using hierarchical numbering + known patterns
    headings = []
    
    print(f"\n🔍 ANALYZING BLOCKS FOR HEADINGS IN {pdf_name}:")
    print("=" * 40)
    
    for idx, row in df.iterrows():
        text = str(row['text']).strip()
        
        # Skip empty or very short text
        if len(text) < 3:
            continue
        
        # Enhanced text quality filtering
        if not is_valid_heading_text(text):
            continue
            
        # Check hierarchical numbering
        is_valid_number, pattern_type, level = False, None, 0
        if has_analyzer:
            is_valid_number, pattern_type, level = analyzer.is_valid_heading_number(text)
        
        # Check exclusion patterns (personal info, etc.)
        is_excluded, exclusion_reason = False, None
        if has_filter:
            is_excluded, exclusion_reason = filter_system.check_exclusion_patterns(text)
        
        # Additional specific exclusions for personal information
        personal_info_patterns = [
            'Junaid Ahmad', 'Registration :', 'B.Tech -', 'Lovely Professional University',
            'Phagwara', 'Computer Science And Engineering'
        ]
        
        for pattern in personal_info_patterns:
            if pattern in text:
                is_excluded = True
                exclusion_reason = f"personal_info_{pattern.replace(' ', '_').replace('.', '').lower()}"
                break
        
        # Check positive patterns
        is_positive, positive_pattern = False, None
        if has_filter:
            is_positive, positive_pattern = filter_system.check_positive_patterns(text)
        
        # Check if it's a known good heading pattern
        is_known_heading = any(pattern in text.upper() for pattern in [
            # Common document sections
            'INTRODUCTION', 'METHODOLOGY', 'CONCLUSION', 'ABSTRACT', 'BACKGROUND',
            'OVERVIEW', 'ANALYSIS', 'IMPLEMENTATION', 'RESULTS', 'DISCUSSION',
            'SUMMARY', 'EXECUTIVE SUMMARY', 'FINDINGS', 'RECOMMENDATIONS',
            'REFERENCES', 'BIBLIOGRAPHY', 'APPENDIX', 'ACKNOWLEDGMENTS',
            'PREFACE', 'FOREWORD', 'TABLE OF CONTENTS', 'INDEX',
            
            # Technical/Academic sections
            'SCOPE', 'PROJECT', 'SYSTEM', 'ARCHITECTURE', 'WORKFLOW', 'STRATEGY',
            'OUTCOMES', 'LEGACY', 'DEPLOYMENT', 'INTEGRATION', 'DEVELOPMENT',
            'CONTAINERIZATION', 'MICROSERVICES', 'PIPELINE', 'FRAMEWORK',
            'DESIGN', 'SPECIFICATION', 'REQUIREMENTS', 'TESTING', 'VALIDATION',
            'EVALUATION', 'PERFORMANCE', 'OPTIMIZATION', 'SECURITY', 'MAINTENANCE',
            'CONFIGURATION', 'INSTALLATION', 'SETUP', 'TROUBLESHOOTING',
            'LITERATURE REVIEW', 'RELATED WORK', 'PRIOR ART', 'STATE OF THE ART',
            
            # Business/Report sections
            'OBJECTIVES', 'GOALS', 'PURPOSE', 'MISSION', 'VISION', 'VALUES',
            'MARKET', 'COMPETITION', 'COMPETITIVE ANALYSIS', 'SWOT',
            'RISK', 'CHALLENGES', 'OPPORTUNITIES', 'TIMELINE', 'SCHEDULE',
            'BUDGET', 'COST', 'RESOURCES', 'TEAM', 'ORGANIZATION',
            'MANAGEMENT', 'GOVERNANCE', 'COMPLIANCE', 'LEGAL', 'REGULATORY',
            'FINANCIAL', 'ECONOMIC', 'IMPACT', 'BENEFITS', 'ROI',
            
            # Chapter/Section indicators
            'CHAPTER', 'SECTION', 'PART', 'UNIT', 'MODULE', 'LESSON',
            'EXERCISE', 'ASSIGNMENT', 'TASK', 'ACTIVITY', 'CASE STUDY',
            'SCENARIO', 'EXAMPLE', 'ILLUSTRATION', 'FIGURE', 'TABLE',
            
            # Research/Academic
            'HYPOTHESIS', 'THEORY', 'MODEL', 'EXPERIMENT', 'SURVEY',
            'INTERVIEW', 'OBSERVATION', 'DATA COLLECTION', 'DATA ANALYSIS',
            'STATISTICAL ANALYSIS', 'QUALITATIVE', 'QUANTITATIVE',
            'METHODS', 'PROCEDURE', 'PROTOCOL', 'ETHICS', 'LIMITATIONS',
            'FUTURE WORK', 'RECOMMENDATIONS', 'IMPLICATIONS',
            
            # Technical documentation
            'API', 'DATABASE', 'NETWORK', 'INFRASTRUCTURE', 'CLOUD',
            'MONITORING', 'LOGGING', 'BACKUP', 'RECOVERY', 'SCALABILITY',
            'AVAILABILITY', 'RELIABILITY', 'USABILITY', 'ACCESSIBILITY',
            'USER INTERFACE', 'USER EXPERIENCE', 'WORKFLOW', 'PROCESS',
            'ALGORITHM', 'DATA STRUCTURE', 'SOFTWARE', 'HARDWARE',
            
            # Problem-solving sections
            'PROBLEM', 'SOLUTION', 'APPROACH', 'ALTERNATIVE', 'OPTION',
            'CRITERIA', 'DECISION', 'SELECTION', 'COMPARISON', 'TRADE-OFF',
            'PROS AND CONS', 'ADVANTAGES', 'DISADVANTAGES', 'ISSUES',
            'CONCERNS', 'MITIGATION', 'CONTINGENCY', 'BACKUP PLAN',
            
            # Quality/Standards
            'QUALITY', 'STANDARDS', 'BEST PRACTICES', 'GUIDELINES',
            'POLICIES', 'PROCEDURES', 'DOCUMENTATION', 'TRAINING',
            'SUPPORT', 'HELP', 'FAQ', 'GLOSSARY', 'TERMS', 'DEFINITIONS'
        ]) and len(text.split()) <= 8  # Must be reasonably short
        
        # Basic manual numbering check as fallback
        if not has_analyzer:
            basic_number_patterns = [
                r'^\d+\.?\s*',           # 1. or 1 
                r'^[IVX]+\.?\s*',        # I. or I
                r'^[A-Z]\.?\s*',         # A. or A
                r'^\d+\.\d+\.?\s*',      # 1.1. or 1.1
                r'^[IVX]+\.[IVX]+\.?\s*' # I.I. or I.I
            ]
            for pattern in basic_number_patterns:
                if re.match(pattern, text):
                    is_valid_number = True
                    pattern_type = "basic_numbering"
                    level = 1  # Default level
                    break
        
        # Decide if it's a heading - Enhanced logic with multiple criteria
        is_heading = False
        reason = "not_heading"
        
        if is_excluded:
            is_heading = False
            reason = f"excluded_{exclusion_reason}"
        elif is_valid_number:
            is_heading = True
            reason = f"numbered_{pattern_type}_level_{level}"
        elif is_positive:
            is_heading = True
            reason = f"positive_{positive_pattern}"
        elif is_known_heading:
            is_heading = True
            reason = "known_heading_pattern"
        else:
            # Enhanced heading detection for non-numbered headings
            clean_text = re.sub(r'[\u0000-\u001F\u007F-\u009F]', '', text).strip()
            word_count = len(clean_text.split())
            has_letters = bool(re.search(r'[a-zA-Z]', clean_text))
            
            # Get text properties from extracted block data
            font_size = row.get('font_size', 12)
            is_bold = row.get('bold', False)
            is_italic = row.get('italic', False) 
            is_all_caps = row.get('is_all_caps', False) or (text.isupper() and has_letters)
            is_title_case = row.get('is_title_case', False)
            ends_with_colon = row.get('ends_with_colon', False)
            starts_with_number = row.get('starts_with_number', False)
            relative_font_size = row.get('relative_font_size', 1.0)
            
            # Calculate text characteristics
            is_short = word_count <= 15  # Increased from 10
            is_medium_short = word_count <= 8
            is_very_short = word_count <= 5
            
            # Check for bad heading indicators (discourse markers, articles, etc.)
            bad_heading_words = ['the ', 'and ', 'or ', 'but ', 'initially', 'these ', 'however', 'therefore', 'thus', 'moreover', 'furthermore']
            has_bad_words = any(bad in text.lower() for bad in bad_heading_words)
            
            # Check if it starts with a capital letter (good heading indicator)
            starts_with_capital = bool(re.match(r'^[A-Z]', clean_text))
            
            # Check title case (multiple capital letters at word boundaries)
            title_case_words = len(re.findall(r'\b[A-Z][a-z]+', text))
            is_title_case = title_case_words >= 2 and word_count <= 10
            
            # Multiple criteria for heading detection
            heading_score = 0
            reasons = []
            
            # Font size criteria (consider relative font size too)
            if font_size > 18 or relative_font_size > 1.5:
                heading_score += 3
                reasons.append("large_font")
            elif font_size > 15 or relative_font_size > 1.3:
                heading_score += 2
                reasons.append("medium_large_font")
            elif font_size > 13 or relative_font_size > 1.1:
                heading_score += 1
                reasons.append("slightly_large_font")
            
            # Bold text (strong heading indicator)
            if is_bold:
                heading_score += 2
                reasons.append("bold")
            
            # Italic text (moderate heading indicator)
            if is_italic and not is_bold:
                heading_score += 1
                reasons.append("italic")
            
            # All caps (strong heading indicator if short)
            if is_all_caps and is_medium_short:
                heading_score += 2
                reasons.append("all_caps")
            elif is_all_caps and is_short:
                heading_score += 1
                reasons.append("all_caps_medium")
            
            # Title case (good heading indicator)
            if is_title_case and is_short:
                heading_score += 2
                reasons.append("title_case")
            elif title_case_words >= 2 and word_count <= 10:
                heading_score += 1
                reasons.append("title_case_manual")
            
            # Ends with colon (section heading indicator)
            if ends_with_colon and is_medium_short:
                heading_score += 2
                reasons.append("ends_with_colon")
            
            # Length criteria
            if is_very_short and not has_bad_words:
                heading_score += 1
                reasons.append("very_short")
            elif is_medium_short and not has_bad_words:
                heading_score += 1
                reasons.append("medium_short")
            
            # Starts with capital (weak but positive indicator)
            if starts_with_capital and not has_bad_words:
                heading_score += 1
                reasons.append("capital_start")
            
            # Position-based criteria (beginning of page or after large gap)
            line_position = row.get('line_position_on_page', 999)
            if line_position <= 0.1:  # Near top of page (first 10%)
                heading_score += 1
                reasons.append("page_top")
            
            # Spacing criteria (if available)
            line_spacing_above = row.get('line_spacing_above')
            if line_spacing_above and line_spacing_above > 20:  # Large gap above
                heading_score += 1
                reasons.append("large_gap_above")
            
            # Penalty for bad heading indicators
            if has_bad_words:
                heading_score -= 2
                reasons.append("bad_words_penalty")
            
            # Check if text is too long to be a heading
            if word_count > 20:
                heading_score -= 2
                reasons.append("too_long_penalty")
            elif word_count > 15:
                heading_score -= 1
                reasons.append("long_penalty")
            
            # Sentence-like text (ends with period and is long)
            if text.endswith('.') and word_count > 10:
                heading_score -= 1
                reasons.append("sentence_like_penalty")
            
            # Bullet points or lists (negative indicator)
            if row.get('bullet', False) or text.strip().startswith(('•', '-', '*', '○', '▪')):
                heading_score -= 2
                reasons.append("bullet_penalty")
            
            # Math or code (negative indicator)
            if row.get('math', False) or '=' in text or any(code_indicator in text for code_indicator in ['()', '{}', '[]', '//']):
                heading_score -= 1
                reasons.append("math_code_penalty")
            
            # Final decision based on score
            if heading_score >= 3 and has_letters and len(clean_text) >= 3:
                is_heading = True
                reason = f"typography_score_{heading_score}_({'+'.join(reasons)})"
            elif (font_size > 15 or is_bold) and is_short and not has_bad_words and has_letters:
                # Fallback for large font or bold headings
                is_heading = True
                reason = "typography_fallback"
        
        # Determine heading level - Enhanced logic
        heading_level = "H3"  # Default
        if is_heading:
            font_size = row.get('font_size', 12)
            is_bold = row.get('bold', False)
            is_all_caps = row.get('is_all_caps', False) or (text.isupper() and bool(re.search(r'[a-zA-Z]', text)))
            word_count = len(text.split())
            relative_font_size = row.get('relative_font_size', 1.0)
            
            if is_valid_number:
                # For numbered headings, use the detected level
                if level == 1:
                    heading_level = "H1"
                elif level == 2:
                    heading_level = "H2"
                else:
                    heading_level = "H3"
            else:
                # For non-numbered headings, use typography and content
                # H1 criteria: Very large font, or large+bold, or short all-caps
                if (font_size >= 22 or relative_font_size >= 1.8 or 
                    (font_size >= 18 and is_bold) or 
                    (font_size >= 16 and is_bold and is_all_caps) or
                    (is_all_caps and word_count <= 3 and font_size >= 14)):
                    heading_level = "H1"
                # H2 criteria: Large font, or medium+bold, or medium all-caps
                elif (font_size >= 16 or relative_font_size >= 1.4 or
                      (font_size >= 14 and is_bold) or 
                      (is_all_caps and word_count <= 6 and font_size >= 12)):
                    heading_level = "H2"
                else:
                    heading_level = "H3"
        
        if is_heading:
            headings.append({
                'text': text,
                'level': heading_level,
                'page': int(row.get('page', 1)),  # Ensure 1-based page numbering
                'reason': reason,
                'font_size': row.get('font_size', 12)
            })
            
            status = "✅ HEADING"
            print(f"{status} | {heading_level} | {text[:50]:<50} | {reason}")
    
    print(f"\n📊 FOUND {len(headings)} HEADINGS")
    
    # Extract document title: first H1 heading within initial 100 blocks
    document_title = extract_document_title(headings, pdf_name)
    
    # Generate JSON structure following output_schema.json
    json_output = {
        "title": document_title,
        "outline": []
    }
    
    for heading in headings:
        json_output["outline"].append({
            "text": heading['text'],
            "level": heading['level'],
            "page": heading['page']
        })
    
    # Save JSON to output folder
    output_path = os.path.join(output_dir, f'{pdf_name}.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)
    
    print(f"✅ JSON SAVED: {output_path}")
    return True

def extract_document_title(headings, pdf_name):
    """Extract document title by finding largest font size text elements at document start"""
    # New approach: Look for the largest font size text elements that form the title
    # This handles multi-line titles with the same large font size
    
    # First try to extract title directly from PDF structure
    try:
        pdf_path = f"input/{pdf_name}.pdf"
        if os.path.exists(pdf_path):
            title_from_pdf = extract_title_from_pdf_structure(pdf_path)
            if title_from_pdf and len(title_from_pdf.strip()) > 5:
                print(f"📖 DOCUMENT TITLE EXTRACTED from PDF structure: '{title_from_pdf}'")
                return title_from_pdf
    except Exception as e:
        print(f"⚠️  Could not extract title from PDF structure: {e}")
    
    # Fallback: Look for first H1 heading within initial 100 blocks
    sorted_headings = sorted(headings, key=lambda h: (h.get('page', 1), headings.index(h)))
    
    # Find the first H1 heading within the first 100 blocks
    for i, heading in enumerate(sorted_headings):
        if i >= 100:  # Only check within initial 100 blocks
            break
            
        if heading.get('level') == 'H1' and heading.get('text', '').strip():
            title = heading['text'].strip()
            print(f"📖 DOCUMENT TITLE EXTRACTED from H1: '{title}' (fallback method)")
            return title
    
    # Final fallback: use PDF filename
    print(f"📖 NO TITLE FOUND, using PDF filename")
    return generate_title_from_pdf_name(pdf_name)

def extract_title_from_pdf_structure(pdf_path):
    """Extract title by analyzing PDF structure for largest font size elements"""
    doc = fitz.open(pdf_path)
    page = doc.load_page(0)  # First page only
    
    # Extract all text blocks with font information
    page_dict = page.get_text('dict')
    text_elements = []
    
    for b in page_dict.get('blocks', []):
        if b.get('type') == 0:  # Text block
            for l in b.get('lines', []):
                for s in l.get('spans', []):
                    text = s.get('text', '').strip()
                    if text and len(text) > 2:
                        text_elements.append({
                            'text': text,
                            'font_size': s.get('size', 0),
                            'bbox': s.get('bbox', []),
                            'y_pos': s.get('bbox', [0, 0, 0, 0])[1]  # y-coordinate for sorting
                        })
    
    if not text_elements:
        doc.close()
        return None
    
    # Sort by y-position (top to bottom)
    text_elements.sort(key=lambda x: x['y_pos'])
    
    # Find the largest font sizes in the first portion of the document
    font_sizes = [elem['font_size'] for elem in text_elements[:20]]  # First 20 elements
    if not font_sizes:
        doc.close()
        return None
    
    max_font_size = max(font_sizes)
    
    # Get all text elements with the maximum font size that appear early in the document
    title_candidates = []
    for elem in text_elements[:15]:  # Look at first 15 elements only
        if elem['font_size'] >= max_font_size - 0.5:  # Allow small font size variation
            # Filter out personal information and unwanted content
            text = elem['text'].strip()
            if not is_unwanted_title_content(text):
                title_candidates.append(text)
    
    doc.close()
    
    if title_candidates:
        # Combine multiple title lines
        title = ' '.join(title_candidates)
        # Clean up the title
        title = ' '.join(title.split())  # Remove extra whitespace
        return title
    
    return None

def is_unwanted_title_content(text):
    """Check if text should be excluded from title (personal info, etc.)"""
    unwanted_patterns = [
        r'^[A-Z][a-z]+\s+[A-Z][a-z]+$',  # Person names like "John Doe"
        r'registration\s*:', 
        r'b\.tech', 
        r'university', 
        r'engineering',
        r'computer\s+science',
        r'phagwara',
        r'\d{6,}',  # Student IDs
        r'^abstract$',
        r'^introduction$',
        r'^conclusion$',
    ]
    
    text_lower = text.lower()
    for pattern in unwanted_patterns:
        if re.search(pattern, text_lower):
            return True
    
    return False

def generate_title_from_pdf_name(pdf_name):
    """Generate a proper title from PDF filename"""
    # Remove common prefixes/suffixes and format as title
    title = pdf_name.replace('_', ' ').replace('-', ' ')
    title = ' '.join(word.capitalize() for word in title.split())
    return title

def main():
    """Main function to process all PDFs in input folder and generate JSON outputs"""
    print("� DIRECT HIERARCHICAL NUMBERING JSON GENERATION")
    print("=" * 60)
    print("📁 Processing all PDFs in input folder")
    print("=" * 60)
    
    # Setup directories
    input_dir = 'input'
    output_dir = 'output'
    
    # Ensure directories exist
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all PDF files in input directory
    pdf_files = list(Path(input_dir).glob("*.pdf"))
    
    if not pdf_files:
        print(f"❌ No PDF files found in {input_dir} directory")
        return False
    
    print(f"� Found {len(pdf_files)} PDF files to process")
    
    successful = 0
    failed = 0
    
    for pdf_path in pdf_files:
        try:
            success = process_single_pdf(str(pdf_path), output_dir)
            if success:
                successful += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Error processing {pdf_path.name}: {e}")
            failed += 1
    
    print(f"\n🎯 PROCESSING COMPLETE!")
    print(f"✅ Successfully processed: {successful}/{len(pdf_files)} PDFs")
    print(f"❌ Failed: {failed}/{len(pdf_files)} PDFs") 
    print(f"📁 JSON files saved to: {output_dir}")
    
    return successful > 0

if __name__ == "__main__":
    main()
