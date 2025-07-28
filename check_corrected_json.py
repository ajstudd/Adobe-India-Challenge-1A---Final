#!/usr/bin/env python3
"""
Check the corrected JSON for numbered headings
"""

import json

def check_corrected_json():
    """Check numbered headings in the corrected JSON"""
    with open('output/Devops_corrected.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    print('🔍 NUMBERED HEADINGS IN CORRECTED JSON:')
    print('=' * 50)

    numbered_patterns = ['I.', 'II.', 'III.', 'IV.', 'V.', 'VI.', 'VII.', 'VIII.', 'IX.', 'X.', 'A.', 'B.', 'C.', 'D.', 'E.', 'F.', 'G.', 'C.1', 'C.2', 'C.3']

    numbered_found = []
    for i, item in enumerate(data['outline']):
        text = item['text']
        level = item['level']
        for pattern in numbered_patterns:
            if text.startswith(pattern):
                numbered_found.append(f'{i+1:2d}: {level} | {text[:60]}')
                break

    for line in numbered_found:
        print(line)
        
    print(f'\nNumbered headings found: {len(numbered_found)}')
    print(f'Total outline items: {len(data["outline"])}')
    
    print('\n❌ CHECKING FOR PERSONAL INFO:')
    print('=' * 50)
    
    personal_patterns = ['Junaid Ahmad', 'Registration :', 'B.Tech -', 'Lovely Professional']
    personal_found = []
    
    for i, item in enumerate(data['outline']):
        text = item['text']
        for pattern in personal_patterns:
            if pattern in text:
                personal_found.append(f'{i+1:2d}: {item["level"]} | {text[:60]}')
                break
    
    if personal_found:
        print("❌ PERSONAL INFO STILL PRESENT:")
        for line in personal_found:
            print(line)
    else:
        print("✅ NO PERSONAL INFO FOUND - CORRECTLY EXCLUDED!")

if __name__ == "__main__":
    check_corrected_json()
