#!/usr/bin/env python3
"""
Quick test of exclusion patterns
"""

import re

# Test the specific patterns I added
test_texts = [
    'Junaid Ahmad',
    'Registration : 12315906', 
    'B.Tech - Computer Science And Engineering',
    'Lovely Professional University',
    'I. Introduction',
    'A. Product Definition',
    'C.1 Docker-Centric Pipeline'
]

# Test exclusion patterns
exclusion_patterns = {
    'specific_names': re.compile(r'^(junaid|ahmad|junaid\s+ahmad)$', re.IGNORECASE),
    'specific_registration': re.compile(r'^registration\s*:\s*12315906$', re.IGNORECASE),
    'specific_course': re.compile(r'^b\.tech\s*-\s*computer\s+science\s+and\s+engineering$', re.IGNORECASE),
    'specific_university': re.compile(r'^lovely\s+professional\s+university$', re.IGNORECASE),
}

print('Testing exclusion patterns:')
for text in test_texts:
    excluded = False
    reason = 'none'
    for pattern_name, pattern in exclusion_patterns.items():
        if pattern.match(text):
            excluded = True
            reason = pattern_name
            break
    
    status = 'EXCLUDED' if excluded else 'ALLOWED'
    print(f'{status:<8} | {text:<45} | Reason: {reason}')
