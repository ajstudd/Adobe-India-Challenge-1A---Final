#!/usr/bin/env python3
"""
Analyze the current training data
"""
import pandas as pd
import glob
import os

def analyze_training_data():
    # Count CSV files
    csv_files = glob.glob('labelled_data/*.csv')
    csv_files = [f for f in csv_files if not f.endswith('_with_predictions.csv')]
    print(f'📄 Found {len(csv_files)} CSV files')

    total_samples = 0
    total_headings = 0
    file_stats = []

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if 'is_heading' in df.columns:
                headings = df['is_heading'].sum()
                percentage = (headings / len(df)) * 100
                total_samples += len(df)
                total_headings += headings
                file_stats.append((os.path.basename(csv_file), len(df), headings, percentage))
        except Exception as e:
            print(f'❌ Error loading {csv_file}: {e}')

    print(f'\n📊 TOTAL DATASET:')
    print(f'   Total samples: {total_samples:,}')
    print(f'   Total headings: {total_headings:,}')
    print(f'   Heading percentage: {(total_headings/total_samples*100):.2f}%')

    print(f'\n📋 FILE BREAKDOWN:')
    for filename, samples, headings, percentage in sorted(file_stats, key=lambda x: x[3], reverse=True):
        print(f'   {filename:<35} {samples:>6} samples, {headings:>4} headings ({percentage:>5.1f}%)')

if __name__ == "__main__":
    analyze_training_data()
