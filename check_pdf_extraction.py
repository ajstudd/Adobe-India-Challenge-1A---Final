#!/usr/bin/env python3
"""
Check PDF extraction for numbered headings
"""

import sys
import os
sys.path.append('.')

from src.extraction.outline_extractor import extract_outline_from_pdf
import pandas as pd

def check_pdf_extraction():
    """Check what's actually being extracted from the PDF"""
    pdf_path = 'input/Devops.pdf'
    
    try:
        print("📄 Extracting outline from PDF...")
        outline = extract_outline_from_pdf(pdf_path)
        
        # Check if outline has the expected structure
        if not outline or 'outline' not in outline:
            print("❌ No outline structure found in PDF")
            return
            
        outline_items = outline['outline']
        print(f'📊 Found {len(outline_items)} outline items')
        print('\n🔍 First 50 outline items:')
        print('=' * 60)
        
        for i, item in enumerate(outline_items[:50]):
            text_clean = str(item.get('text', '')).strip()[:60]
            level = item.get('level', 'N/A')
            page = item.get('page', 'N/A')
            print(f'{i:2d}: {level:<3} | Page {page} | {text_clean}')
            
        print('\n🔍 Looking for numbered patterns...')
        numbered_patterns = []
        for i, item in enumerate(outline_items):
            text_clean = str(item.get('text', '')).strip()
            # Look for patterns like I., II., A., B., 1., 1.1, etc.
            if any(pattern in text_clean[:10] for pattern in ['I.', 'II.', 'III.', 'IV.', 'V.', 'VI.', 'VII.', 'VIII.', 'A.', 'B.', 'C.', 'D.', 'E.', '1.', '2.', '3.', '1.1', '1.2', 'C.1', 'C.2', 'C.3']):
                numbered_patterns.append((i, text_clean[:80], item.get('level', 'N/A')))
        
        print(f'\n📊 Found {len(numbered_patterns)} potential numbered patterns:')
        for idx, text, level in numbered_patterns:
            print(f'  {idx:3d}: {level:<3} | {text}')
            
        # Also check for roman numerals in isolation
        print('\n🏛️ Looking for standalone roman numerals...')
        roman_patterns = []
        for i, item in enumerate(outline_items):
            text_clean = str(item.get('text', '')).strip()
            if text_clean in ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X']:
                roman_patterns.append((i, text_clean, item.get('level', 'N/A')))
        
        print(f'📊 Found {len(roman_patterns)} standalone roman numerals:')
        for idx, text, level in roman_patterns:
            print(f'  {idx:3d}: {level:<3} | {text}')
            
    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_pdf_extraction()
