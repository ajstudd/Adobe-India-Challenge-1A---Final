#!/usr/bin/env python3
"""
Test script for enhanced intelligent filter with hierarchical numbering
"""

from intelligent_filter import IntelligentFilter
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def test_hierarchical_numbering():
    """Test the enhanced filter with hierarchical numbering patterns"""
    print("🧪 TESTING ENHANCED INTELLIGENT FILTER")
    print("=" * 50)
    
    # Test the enhanced filter
    filter_system = IntelligentFilter()
    
    # Test data with hierarchical numbering - same patterns as in your JSON
    test_data = {
        'text': [
            'I. Introduction',
            'II. Methodology', 
            'A. Product Definition',
            'B. Feasibility Analysis',
            'C. Continuous Integration and Deployment',
            '1. Overview',
            '1.1 Background',
            '1.1.1 Details',
            'A.1 Mixed Example',
            'Problem Analysis',
            ', 2024. Available:',
            'Bcrypt.js',
            'Docker-based',
            'Initially,',
            'VIII. Outcomes',
            'C.1 Docker-Centric Pipeline',
            'C.2 Jenkins Integration and Security Enforcement',
            'C.3 Frontend Deployment Workflow',
            'D. Deployment Strategy',
            'E. Oracle VPS-Based Deployment Strategy'
        ],
        'font_size': [14, 14, 13, 13, 13, 12, 11, 10, 12, 13, 10, 10, 11, 10, 14, 12, 12, 12, 13, 13],
        'page': [1, 1, 2, 2, 4, 2, 2, 2, 3, 2, 7, 2, 2, 3, 5, 4, 4, 4, 4, 6],
        'line_position_on_page': [1, 5, 8, 12, 15, 18, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85],
        'is_heading_pred': [1] * 20,
        'heading_confidence': [0.9, 0.9, 0.85, 0.85, 0.85, 0.8, 0.75, 0.7, 0.8, 0.6, 0.4, 0.5, 0.3, 0.2, 0.95, 0.8, 0.8, 0.8, 0.85, 0.85]
    }
    
    df = pd.DataFrame(test_data)
    print(f"📊 Testing {len(df)} text samples...")
    
    # Apply filtering
    filtered_df = filter_system.apply_intelligent_filtering(df)
    
    # Debug: Print column names to see what's available
    print(f"\n🔍 DEBUG - Available columns: {list(filtered_df.columns)}")
    print(f"🔍 DEBUG - Sample row keys: {list(filtered_df.iloc[0].keys()) if len(filtered_df) > 0 else 'No data'}")
    
    print("\n📈 FILTERING RESULTS:")
    print("=" * 40)
    
    for idx, row in filtered_df.iterrows():
        text = row['text']
        decision = row.get('filter_decision', 'N/A')
        confidence = row.get('heading_confidence', 0)
        filter_score = row.get('filter_score', 0)
        
        # Check if it would be in the final output - try multiple possible column names
        is_heading = row.get('is_heading_final', 
                           row.get('is_heading_pred_final', 
                                  row.get('is_heading_pred', 0)))
        
        # Also check filter decision for keep/filter determination
        is_kept_by_decision = 'keep' in decision.lower() if decision != 'N/A' else False
        final_status = is_heading == 1 or is_kept_by_decision
        
        status = "✅ KEPT" if final_status else "❌ FILTERED"
        print(f"{status} | '{text[:50]:<50}' | Conf: {confidence:.2f} | Score: {filter_score:.2f} | Decision: {decision}")
    
    # Summary - fix the counting logic
    kept_count = sum(1 for _, row in filtered_df.iterrows() 
                    if (row.get('is_heading_final', row.get('is_heading_pred_final', row.get('is_heading_pred', 0))) == 1 
                        or 'keep' in row.get('filter_decision', '').lower()))
    total_count = len(filtered_df)
    
    print(f"\n📊 SUMMARY:")
    print(f"   Total texts: {total_count}")
    print(f"   Kept as headings: {kept_count}")
    print(f"   Filtered out: {total_count - kept_count}")
    print(f"   Retention rate: {(kept_count/total_count)*100:.1f}%")
    
    # Check specific patterns
    print(f"\n🔍 PATTERN ANALYSIS:")
    hierarchical_patterns = ['I.', 'II.', 'A.', 'B.', 'C.', '1.', '1.1', '1.1.1', 'A.1', 'VIII.', 'C.1', 'C.2', 'C.3', 'D.', 'E.']
    
    for pattern in hierarchical_patterns:
        matching_rows = filtered_df[filtered_df['text'].str.startswith(pattern)]
        if len(matching_rows) > 0:
            for _, row in matching_rows.iterrows():
                is_kept = (row.get('is_heading_final', row.get('is_heading_pred_final', row.get('is_heading_pred', 0))) == 1 
                          or 'keep' in row.get('filter_decision', '').lower())
                status = "✅ KEPT" if is_kept else "❌ FILTERED"
                print(f"   {pattern:<6} pattern: {status} - '{row['text']}'")

if __name__ == "__main__":
    test_hierarchical_numbering()
