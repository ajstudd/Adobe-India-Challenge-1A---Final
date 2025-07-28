#!/usr/bin/env python3
"""
Dataset Quality Improvement Tool
===============================

This tool helps identify which files in your dataset are most valuable
for training and suggests improvements.
"""

import pandas as pd
import glob
import os
import numpy as np

def analyze_dataset_quality():
    """Analyze dataset quality and suggest improvements"""
    print("📊 DATASET QUALITY ANALYSIS")
    print("=" * 50)
    
    csv_files = glob.glob('labelled_data/*.csv')
    csv_files = [f for f in csv_files if not f.endswith('_with_predictions.csv')]
    
    file_analysis = []
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if 'is_heading' in df.columns:
                analysis = analyze_single_file(df, csv_file)
                file_analysis.append(analysis)
        except Exception as e:
            print(f"❌ Error analyzing {csv_file}: {e}")
    
    # Sort by quality score
    file_analysis.sort(key=lambda x: x['quality_score'], reverse=True)
    
    print(f"\n📋 FILE QUALITY RANKING:")
    print("=" * 70)
    print(f"{'Rank':<4} {'File':<35} {'Score':<6} {'Headings':<9} {'Samples':<8} {'%':<6}")
    print("-" * 70)
    
    high_quality = []
    medium_quality = []
    low_quality = []
    
    for i, analysis in enumerate(file_analysis, 1):
        filename = analysis['filename']
        score = analysis['quality_score']
        headings = analysis['headings']
        samples = analysis['samples']
        percentage = analysis['percentage']
        
        print(f"{i:<4} {filename:<35} {score:<6.1f} {headings:<9} {samples:<8} {percentage:<6.1f}")
        
        if score >= 7.0:
            high_quality.append(analysis)
        elif score >= 4.0:
            medium_quality.append(analysis)
        else:
            low_quality.append(analysis)
    
    print("-" * 70)
    print(f"📊 QUALITY SUMMARY:")
    print(f"   🟢 High Quality (≥7.0): {len(high_quality)} files")
    print(f"   🟡 Medium Quality (4.0-6.9): {len(medium_quality)} files")
    print(f"   🔴 Low Quality (<4.0): {len(low_quality)} files")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print("-" * 20)
    
    if len(high_quality) < 5:
        print("🎯 PRIORITY: Need more high-quality files")
        print("   - Look for documents with clear heading hierarchy")
        print("   - Academic papers, technical manuals, reports work best")
    
    if len(low_quality) > len(high_quality):
        print("⚠️  WARNING: Too many low-quality files")
        print("   - Consider removing files with <1% headings")
        print("   - Focus on improving data quality over quantity")
    
    # Calculate total effective samples
    total_high_samples = sum(f['samples'] for f in high_quality)
    total_high_headings = sum(f['headings'] for f in high_quality)
    
    if total_high_samples > 0:
        effective_percentage = (total_high_headings / total_high_samples) * 100
        print(f"\n📈 EFFECTIVE DATASET (High Quality Files Only):")
        print(f"   Samples: {total_high_samples:,}")
        print(f"   Headings: {total_high_headings:,}")
        print(f"   Percentage: {effective_percentage:.2f}%")
        
        if effective_percentage > 5.0:
            print("   ✅ Good heading density for training")
        else:
            print("   ⚠️  Still low heading density")
    
    # Suggest specific improvements
    print(f"\n🔧 SPECIFIC IMPROVEMENTS:")
    print("-" * 25)
    
    if low_quality:
        print("1. 🗑️  Consider removing these low-quality files:")
        for analysis in low_quality[:5]:  # Show top 5 worst
            print(f"   - {analysis['filename']} ({analysis['percentage']:.1f}% headings)")
    
    if len(high_quality) > 0:
        print("2. ✅ Keep and prioritize these high-quality files:")
        for analysis in high_quality[:3]:  # Show top 3 best
            print(f"   - {analysis['filename']} ({analysis['percentage']:.1f}% headings)")
    
    print("3. 📝 To improve dataset:")
    print("   - Add more documents with obvious heading structure")
    print("   - Look for files with H1, H2, H3 style headings")
    print("   - Academic papers and technical manuals are ideal")

def analyze_single_file(df, filepath):
    """Analyze quality of a single CSV file"""
    filename = os.path.basename(filepath)
    
    total_samples = len(df)
    headings = df['is_heading'].sum()
    percentage = (headings / total_samples) * 100
    
    # Calculate quality score (0-10)
    quality_score = 0
    
    # Heading percentage score (0-4 points)
    if percentage >= 8.0:
        quality_score += 4
    elif percentage >= 5.0:
        quality_score += 3
    elif percentage >= 2.0:
        quality_score += 2
    elif percentage >= 1.0:
        quality_score += 1
    
    # Sample size score (0-2 points)
    if total_samples >= 1000:
        quality_score += 2
    elif total_samples >= 500:
        quality_score += 1
    
    # Heading count score (0-2 points)
    if headings >= 50:
        quality_score += 2
    elif headings >= 20:
        quality_score += 1
    
    # Data completeness score (0-2 points) - handle missing columns gracefully
    required_cols = ['text', 'font_size', 'page', 'x', 'y']
    complete_cols = 0
    
    for col in required_cols:
        if col in df.columns:
            # Check if column exists and has non-null values
            non_null_count = df[col].notna().sum()
            if non_null_count > 0:
                complete_cols += 1
        elif col in ['page'] and 'page_num' in df.columns:
            # Handle alternative column names
            non_null_count = df['page_num'].notna().sum()
            if non_null_count > 0:
                complete_cols += 1
    
    # Award partial points for data completeness
    quality_score += (complete_cols / len(required_cols)) * 2
    
    return {
        'filename': filename,
        'filepath': filepath,
        'samples': total_samples,
        'headings': headings,
        'percentage': percentage,
        'quality_score': quality_score
    }

if __name__ == "__main__":
    analyze_dataset_quality()
