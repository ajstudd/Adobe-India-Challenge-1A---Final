#!/usr/bin/env python3
"""
Simple test for merge and title detection
"""

import pandas as pd
from intelligent_filter import IntelligentFilter

# Create simple test data
test_data = {
    'text': [
        'PROJECT TITLE',  # Should be title (H1)
        '1',  # Should merge with next
        'Introduction',  # Should be merged from previous
        'Some paragraph.',
        'A.',  # Should merge with next
        'System Design',  # Should be merged from previous
    ],
    'font_size': [16, 12, 14, 11, 12, 14],
    'page': [1, 1, 1, 1, 1, 1],
    'line_position_on_page': [1, 5, 6, 10, 15, 16],
    'is_heading_pred': [1, 1, 1, 0, 1, 1],
    'heading_confidence': [0.9, 0.7, 0.8, 0.2, 0.7, 0.8],
    'x0': [50] * 6,
    'y0': [750, 650, 630, 600, 550, 530],
    'x1': [200] * 6,
    'y1': [770, 665, 645, 615, 565, 545]
}

df = pd.DataFrame(test_data)
print(f"Original headings: {df['is_heading_pred'].sum()}")

# Test filtering
filter_system = IntelligentFilter()

try:
    result = filter_system.apply_intelligent_filtering(df)
    print(f"Final headings: {result['is_heading_pred'].sum()}")
    
    # Show results
    for idx, row in result.iterrows():
        if row['is_heading_pred'] == 1:
            level = row.get('final_heading_level', 'H3')
            decision = row.get('filter_decision', 'unknown')
            print(f"{level}: '{row['text']}' ({decision})")
            
    print("\n=== Testing completed successfully! ===")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
