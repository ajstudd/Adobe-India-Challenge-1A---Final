#!/usr/bin/env python3
"""
Check CSV data for numbered headings and personal info
"""

import pandas as pd

def check_csv_data():
    """Check the is_heading values in the CSV"""
    df = pd.read_csv('output/training_data/Devops_blocks.csv')

    # Find numbered headings that should be kept
    numbered_patterns = ['I.', 'II.', 'III.', 'IV.', 'V.', 'VI.', 'VII.', 'VIII.', 'A.', 'B.', 'C.', 'D.', 'E.', '1.', '1.1', 'C.1', 'C.2', 'C.3']
    print('NUMBERED HEADINGS IN CSV:')
    print('=' * 50)

    for idx, row in df.iterrows():
        text = str(row['text']).strip()
        for pattern in numbered_patterns:
            if text.startswith(pattern):
                heading_value = row['is_heading']
                print(f'{idx:3d}: {heading_value} | {text[:70]}')
                break

    print()
    print('PERSONAL INFO IN CSV:')
    print('=' * 50)

    personal_patterns = ['Junaid Ahmad', 'Registration :', 'B.Tech -', 'Lovely Professional']
    for idx, row in df.iterrows():
        text = str(row['text']).strip()
        for pattern in personal_patterns:
            if pattern in text:
                heading_value = row['is_heading']
                print(f'{idx:3d}: {heading_value} | {text[:70]}')
                break

    print()
    print('ALL HEADINGS (is_heading=1):')
    print('=' * 50)
    
    headings = df[df['is_heading'] == 1]
    for idx, row in headings.iterrows():
        text = str(row['text']).strip()
        print(f'{idx:3d}: {text[:70]}')

if __name__ == "__main__":
    check_csv_data()
