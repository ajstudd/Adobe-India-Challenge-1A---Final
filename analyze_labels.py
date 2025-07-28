#!/usr/bin/env python3
import pandas as pd
import glob
import os

# Analyze labeled data distribution
labelled_data_dir = "labelled_data"
csv_files = glob.glob(os.path.join(labelled_data_dir, "*.csv"))
csv_files = [f for f in csv_files if not f.endswith('_with_predictions.csv')]

total_headings = 0
total_rows = 0
file_stats = []

print("Analyzing labeled data distribution:")
print("=" * 60)

for csv_file in csv_files:
    filename = os.path.basename(csv_file)
    try:
        df = pd.read_csv(csv_file)
        heading_count = df['is_heading'].sum()
        row_count = len(df)
        total_headings += heading_count
        total_rows += row_count
        
        heading_percentage = heading_count/row_count*100
        file_stats.append({
            'file': filename,
            'headings': heading_count,
            'total': row_count,
            'percentage': heading_percentage
        })
        
        print(f"{filename:<35} {heading_count:>4}/{row_count:>4} headings ({heading_percentage:>5.1f}%)")
        
        # Check heading levels
        if 'heading_level' in df.columns:
            level_counts = df[df['is_heading']==1]['heading_level'].value_counts().sort_index()
            if not level_counts.empty:
                level_str = ", ".join([f"H{int(level)}:{count}" for level, count in level_counts.items() if pd.notna(level)])
                print(f"{'':>37} Levels: {level_str}")
        
    except Exception as e:
        print(f"{filename:<35} ERROR: {e}")

print("=" * 60)
total_percentage = total_headings/total_rows*100
print(f"{'TOTAL':<35} {total_headings:>4}/{total_rows:>4} headings ({total_percentage:>5.1f}%)")

print(f"\nSummary:")
print(f"- Total files: {len(csv_files)}")
print(f"- Average heading percentage: {total_percentage:.1f}%")
print(f"- Files with low heading %: {len([f for f in file_stats if f['percentage'] < 2.0])}")
print(f"- Files with good heading %: {len([f for f in file_stats if f['percentage'] >= 2.0])}")
