#!/usr/bin/env python3
"""
Simple JSON regeneration with enhanced filtering
"""

import pandas as pd
import json
from datetime import datetime
import sys
import os

sys.path.append('.')

def apply_enhanced_filtering():
    """Apply enhanced filtering to existing CSV data"""
    print("🧠 APPLYING ENHANCED FILTERING TO EXISTING DATA")
    print("=" * 60)
    
    # Load existing CSV data
    csv_path = 'output/training_data/Devops_blocks.csv'
    if not os.path.exists(csv_path):
        print(f"❌ File not found: {csv_path}")
        return
    
    print(f"📊 Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"📈 Loaded {len(df)} rows")
    
    # Apply intelligent filtering
    try:
        from intelligent_filter import IntelligentFilter
        filter_system = IntelligentFilter()
        
        print("🧠 Applying enhanced intelligent filtering...")
        filtered_df = filter_system.apply_intelligent_filtering(df, confidence_col='heading_confidence')
        
        print(f"📊 Filtering completed!")
        print(f"📈 Original predictions: {df['is_heading_pred'].sum()}")
        print(f"📈 After filtering: {filtered_df['is_heading_pred'].sum()}")
        
        # Copy the filtered results
        result_df = filtered_df.copy()
        result_df['is_heading'] = result_df['is_heading_pred']
        
        # Generate JSON output
        print("🔧 Generating JSON output...")
        json_output = generate_json_from_dataframe(result_df)
        
        # Save JSON output
        output_path = 'output/Devops.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_output, f, indent=2, ensure_ascii=False)
        
        print(f"✅ JSON saved to: {output_path}")
        
        # Show preview of headings found
        headings = result_df[result_df['is_heading'] == 1]
        print(f"\n📋 HEADINGS FOUND ({len(headings)} total):")
        print("=" * 50)
        
        for idx, row in headings.head(20).iterrows():
            text = str(row['text'])[:60]
            level = row.get('heading_level', 'N/A')
            confidence = row.get('heading_confidence', 0)
            print(f"  {level:<3} | {confidence:>5.2f} | {text}")
        
        if len(headings) > 20:
            print(f"  ... and {len(headings) - 20} more headings")
            
    except Exception as e:
        print(f"❌ Error during filtering: {e}")
        import traceback
        traceback.print_exc()

def generate_json_from_dataframe(df):
    """Generate JSON structure from filtered dataframe"""
    
    # Get headings only
    headings_df = df[df['is_heading'] == 1].copy()
    
    # Basic JSON structure
    json_output = {
        "document_title": "Implementation of DevOps Practices in Proactive India",
        "outline": []
    }
    
    for idx, row in headings_df.iterrows():
        heading_item = {
            "text": str(row['text']).strip(),
            "level": str(row.get('heading_level', 'H1')),
            "page": int(row.get('page', 1)) if pd.notna(row.get('page')) else 1,
            "bbox": {
                "x1": float(row.get('x1', 0)) if pd.notna(row.get('x1')) else 0,
                "y1": float(row.get('y1', 0)) if pd.notna(row.get('y1')) else 0,
                "x2": float(row.get('x2', 100)) if pd.notna(row.get('x2')) else 100,
                "y2": float(row.get('y2', 20)) if pd.notna(row.get('y2')) else 20
            }
        }
        json_output["outline"].append(heading_item)
    
    return json_output

if __name__ == "__main__":
    apply_enhanced_filtering()
