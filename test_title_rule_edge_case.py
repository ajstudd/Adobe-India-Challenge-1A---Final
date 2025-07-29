#!/usr/bin/env python3
"""
Test script to verify the document title rule edge case - no heading in first 100 blocks
"""
import pandas as pd
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from intelligent_filter import IntelligentFilter

def test_no_title_in_100_blocks():
    """Test the document title rule when first heading is beyond 100 blocks"""
    print("🧪 Testing Document Title Rule - No H1 in first 100 blocks")
    print("=" * 60)
    
    # Create test data with first heading at position 105 (beyond 100 blocks)
    test_data = []
    for i in range(120):
        if i == 105:  # First heading at position 105 (beyond 100 blocks)
            test_data.append({
                'text': 'Late Document Title',
                'page': 2,
                'line_position_on_page': i,
                'is_heading_pred': 1,
                'heading_confidence': 0.9,
                'final_heading_level': 'H3'
            })
        elif i == 110:  # Second heading
            test_data.append({
                'text': 'Another Late Heading',
                'page': 2,
                'line_position_on_page': i,
                'is_heading_pred': 1,
                'heading_confidence': 0.8,
                'final_heading_level': 'H3'
            })
        else:
            test_data.append({
                'text': f'Regular paragraph text block {i}',
                'page': 1 if i < 100 else 2,
                'line_position_on_page': i,
                'is_heading_pred': 0,
                'heading_confidence': 0.2,
                'final_heading_level': 'H3'
            })

    df = pd.DataFrame(test_data)
    print(f"✅ Created test dataframe with {len(df)} rows")
    print(f"   📍 First heading at position 105 (beyond first 100 blocks)")
    
    # Initialize filter and test
    print("🔧 Initializing IntelligentFilter...")
    filter_system = IntelligentFilter()
    
    print("📄 Applying document title rule...")
    result_df = filter_system.apply_document_title_rule(df)
    
    # Check results
    if 'filter_decision' in result_df.columns:
        title_rows = result_df[result_df['filter_decision'].fillna('') == 'document_title']
        print(f"\n📊 Results:")
        print(f"   🔢 Found {len(title_rows)} document titles")
        
        if len(title_rows) == 0:
            print("   ✅ PASS: No document title identified (as expected)")
            print("   ✅ PASS: Rule correctly ignores headings beyond first 100 blocks")
        else:
            title_row = title_rows.iloc[0]
            print(f"   ❌ FAIL: Document title incorrectly identified: \"{title_row['text']}\"")
            print(f"   📍 Position: {title_rows.index[0]} (line {title_row['line_position_on_page']})")
    else:
        print(f"\n📊 Results:")
        print(f"   🔢 Found 0 document titles")
        print("   ✅ PASS: No document title identified (as expected)")
        print("   ✅ PASS: Rule correctly ignores headings beyond first 100 blocks")

if __name__ == "__main__":
    test_no_title_in_100_blocks()
