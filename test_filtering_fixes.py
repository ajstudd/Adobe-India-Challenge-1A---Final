#!/usr/bin/env python3
"""
Test script for the fixed intelligent filtering system
"""

import os
import pandas as pd
import numpy as np
from intelligent_filter import IntelligentFilter

def test_merge_and_title_detection():
    """Test the merge feature and title detection"""
    print("🧪 Testing Merge Feature and Title Detection")
    print("=" * 50)
    
    # Create test data that should trigger merge and title detection
    test_data = {
        'text': [
            'DEVOPS PROJECT REPORT',  # Should be detected as title (H1)
            'Computer Science Department',  # Should be subtitle (H2)
            '1',  # Should merge with next
            'Introduction',  # Should be merged from previous
            'This is a regular paragraph about the project.',
            'A.',  # Should merge with next
            'System Architecture',  # Should be merged from previous
            'The system uses various technologies.',
            'II',  # Should merge with next
            'Methodology and Implementation',  # Should be merged from previous
            'Details about implementation follow.',
            '2.1',  # Should merge with next
            'Database Design',  # Should be merged from previous
            'References'  # Should be heading
        ],
        'font_size': [18, 14, 12, 14, 11, 12, 14, 11, 12, 14, 11, 12, 13, 13],
        'page': [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3],
        'line_position_on_page': [1, 3, 8, 9, 15, 20, 21, 25, 2, 3, 8, 12, 13, 5],
        'is_heading_pred': [1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1],
        'heading_confidence': [0.95, 0.85, 0.75, 0.80, 0.2, 0.70, 0.85, 0.15, 0.65, 0.90, 0.1, 0.60, 0.80, 0.85],
        'x0': [50] * 14,
        'y0': [750, 720, 650, 630, 600, 550, 530, 500, 750, 730, 680, 600, 580, 400],
        'x1': [200] * 14,
        'y1': [770, 735, 665, 645, 615, 565, 545, 515, 765, 745, 695, 615, 595, 415]
    }
    
    df = pd.DataFrame(test_data)
    print(f"📊 Original data: {len(df)} rows")
    print(f"📊 Original headings: {df['is_heading_pred'].sum()}")
    
    # Initialize the filter
    filter_system = IntelligentFilter()
    
    try:
        # Apply filtering
        print("\n🔧 Applying intelligent filtering...")
        filtered_df = filter_system.apply_intelligent_filtering(df)
        
        print(f"\n📊 Results after filtering:")
        print(f"   Total rows: {len(filtered_df)}")
        print(f"   Final headings: {filtered_df['is_heading_pred'].sum()}")
        
        # Check merge results
        print(f"\n🔗 Checking merge results:")
        merged_blocks = filtered_df[filtered_df['filter_decision'].str.contains('merged', na=False)]
        print(f"   Merged blocks: {len(merged_blocks)}")
        
        for idx, row in merged_blocks.iterrows():
            print(f"   - '{row['text']}' ({row['filter_decision']})")
        
        # Check title detection
        print(f"\n📄 Checking title detection:")
        title_blocks = filtered_df[filtered_df['filter_decision'].str.contains('title|subtitle', na=False)]
        print(f"   Title/subtitle blocks: {len(title_blocks)}")
        
        for idx, row in title_blocks.iterrows():
            level = row.get('final_heading_level', 'Unknown')
            print(f"   - '{row['text']}' ({level}) - {row['filter_decision']}")
        
        # Show final headings
        print(f"\n📋 Final heading structure:")
        final_headings = filtered_df[filtered_df['is_heading_pred'] == 1]
        for idx, row in final_headings.iterrows():
            level = row.get('final_heading_level', 'H3')
            decision = row.get('filter_decision', 'unknown')
            print(f"   {level}: '{row['text']}' ({decision})")
        
        return filtered_df
        
    except Exception as e:
        print(f"❌ Error during filtering: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_edge_cases():
    """Test edge cases for merging"""
    print(f"\n🧪 Testing Edge Cases")
    print("=" * 30)
    
    edge_data = {
        'text': [
            '1',  # Should merge
            'Chapter Introduction',  # Should be merged target
            '2',  # Should merge
            'a',  # Should NOT merge (lowercase)
            'III',  # Should merge
            'Background and Related Work',  # Should be merged target
            'Some paragraph text here.',
            'A.',  # Should merge
            'System Overview',  # Should be merged target
        ],
        'font_size': [12, 14, 12, 11, 12, 14, 11, 12, 14],
        'page': [1] * 9,
        'line_position_on_page': list(range(1, 10)),
        'is_heading_pred': [1, 1, 1, 1, 1, 1, 0, 1, 1],
        'heading_confidence': [0.7, 0.8, 0.7, 0.6, 0.7, 0.85, 0.2, 0.75, 0.8],
        'x0': [50] * 9,
        'y0': list(range(750, 650, -10)),
        'x1': [200] * 9,
        'y1': list(range(765, 665, -10))
    }
    
    df = pd.DataFrame(edge_data)
    filter_system = IntelligentFilter()
    
    try:
        filtered_df = filter_system.apply_intelligent_filtering(df)
        
        print(f"📊 Edge case results:")
        merged_or_removed = filtered_df[
            (filtered_df['filter_decision'].str.contains('merged', na=False)) |
            (filtered_df['is_heading_pred'] == 0)
        ]
        
        for idx, row in merged_or_removed.iterrows():
            original_pred = df.loc[idx, 'is_heading_pred']
            current_pred = row['is_heading_pred']
            status = "MERGED" if 'merged' in str(row['filter_decision']) else "FILTERED"
            print(f"   {status}: '{row['text']}' (was: {original_pred}, now: {current_pred})")
        
        return filtered_df
        
    except Exception as e:
        print(f"❌ Error in edge case testing: {e}")
        return None

if __name__ == "__main__":
    print("🚀 TESTING FIXED INTELLIGENT FILTERING SYSTEM")
    print("=" * 60)
    
    # Test 1: Main functionality
    result1 = test_merge_and_title_detection()
    
    # Test 2: Edge cases
    result2 = test_edge_cases()
    
    if result1 is not None and result2 is not None:
        print(f"\n✅ All tests completed successfully!")
    else:
        print(f"\n❌ Some tests failed!")
