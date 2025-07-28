#!/usr/bin/env python3
"""
Test Enhanced Intelligent Filtering
===================================

This script tests the enhanced intelligent filtering system with specific 
false positive examples from the requirements document.

Author: AI Assistant
Date: July 28, 2025
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from intelligent_filter import IntelligentFilter

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_data():
    """Create test data with known false positives from the requirements"""
    
    # Examples of incorrect headings from the requirements document
    false_positives = [
        "Junaid Ahmad",
        "Registration : 12315906", 
        "Lovely Professional University",
        "Phagwara",
        "Bcrypt.js",
        "These",
        "Initially,",
        "The system's containerized infrastructure, powered by Docker",
        "automation, and Oracle VPS as the hosting environment. This"
    ]
    
    # Examples of correct headings
    true_positives = [
        "Chapter 1: Introduction",
        "1.1 Overview",
        "Methodology", 
        "Results and Discussion",
        "Conclusion",
        "References",
        "2. Literature Review",
        "3.1 System Architecture"
    ]
    
    # Create test dataframe
    test_data = []
    
    # Add false positives (should be filtered out)
    for i, text in enumerate(false_positives):
        test_data.append({
            'text': text,
            'font_size': 12 + np.random.randint(-2, 3),  # Random font sizes around 12
            'page': 1,
            'x0': 50,
            'y0': 100 + i * 20,
            'x1': 200,
            'y1': 115 + i * 20,
            'line_position_on_page': 5 + i,
            'word_count': len(text.split()),
            'is_heading_pred': 1,  # ML model predicted as heading
            'heading_confidence': 0.7 + np.random.random() * 0.2,  # Random confidence 0.7-0.9
            'bold': False,
            'italic': False,
            'is_heading': 0,  # Ground truth: not a heading
            'num_nouns': max(1, len([w for w in text.split() if w.isalpha()])),
            'num_verbs': 0,  # Most false positives have no verbs
            'num_adjs': 0,
            'num_advs': 0
        })
    
    # Add true positives (should be kept)
    for i, text in enumerate(true_positives):
        test_data.append({
            'text': text,
            'font_size': 14 + np.random.randint(-1, 3),  # Slightly larger fonts
            'page': 1,
            'x0': 50,
            'y0': 300 + i * 30,
            'x1': 250,
            'y1': 320 + i * 30,
            'line_position_on_page': 1 + i * 3,  # More spread out
            'word_count': len(text.split()),
            'is_heading_pred': 1,  # ML model predicted as heading
            'heading_confidence': 0.8 + np.random.random() * 0.15,  # Higher confidence
            'bold': True,  # Many headings are bold
            'italic': False,
            'is_heading': 1,  # Ground truth: is a heading
            'num_nouns': max(1, len([w for w in text.split() if w.isalpha()])),
            'num_verbs': min(1, len([w for w in text.split() if w.lower() in ['review', 'discussion']])),
            'num_adjs': 0,
            'num_advs': 0
        })
    
    # Add some non-heading text (correctly predicted as non-heading)
    non_headings = [
        "This is a regular paragraph with normal text that describes the methodology used in this study.",
        "The results show that the proposed approach achieves better performance than existing methods.",
        "Figure 1 shows the system architecture diagram.",
        "According to the literature review, several approaches have been proposed for this problem."
    ]
    
    for i, text in enumerate(non_headings):
        test_data.append({
            'text': text,
            'font_size': 11,  # Smaller font
            'page': 1,
            'x0': 50,
            'y0': 600 + i * 25,
            'x1': 400,
            'y1': 615 + i * 25,
            'line_position_on_page': 10 + i * 2,
            'word_count': len(text.split()),
            'is_heading_pred': 0,  # Correctly predicted as non-heading
            'heading_confidence': 0.2 + np.random.random() * 0.3,  # Low confidence
            'bold': False,
            'italic': False,
            'is_heading': 0,  # Ground truth: not a heading
            'num_nouns': max(2, len([w for w in text.split() if w.isalpha()]) // 2),
            'num_verbs': max(1, len([w for w in text.split() if w.lower() in ['show', 'used', 'achieves', 'proposed']])),
            'num_adjs': 1,
            'num_advs': 0
        })
    
    return pd.DataFrame(test_data)

def test_filtering_performance():
    """Test the filtering performance on known examples"""
    
    logger.info("🧪 Testing Enhanced Intelligent Filtering System")
    logger.info("=" * 60)
    
    # Create test data
    test_df = create_test_data()
    logger.info(f"📊 Created test dataset with {len(test_df)} samples")
    
    # Debug: Show all predictions
    all_predictions = test_df[test_df['is_heading_pred'] == 1]
    logger.info(f"   - All ML predictions: {len(all_predictions)}")
    
    for _, row in all_predictions.iterrows():
        truth = "TRUE HEADING" if row['is_heading'] == 1 else "FALSE POSITIVE"
        logger.info(f"     * '{row['text']}' -> {truth}")
    
    false_positives_count = len(test_df[(test_df['is_heading_pred'] == 1) & (test_df['is_heading'] == 0)])
    true_positives_count = len(test_df[(test_df['is_heading_pred'] == 1) & (test_df['is_heading'] == 1)])
    not_predicted_count = len(test_df[test_df['is_heading_pred'] == 0])
    
    logger.info(f"   - False positives (should be filtered): {false_positives_count}")
    logger.info(f"   - True positives (should be kept): {true_positives_count}")
    logger.info(f"   - True negatives (not predicted): {not_predicted_count}")
    
    # Initialize the intelligent filter
    filter_system = IntelligentFilter()
    
    # Apply filtering
    logger.info("\n🧠 Applying intelligent filtering...")
    filtered_df = filter_system.apply_intelligent_filtering(test_df, confidence_col='heading_confidence')
    
    # Analyze results
    logger.info("\n📊 FILTERING ANALYSIS")
    logger.info("=" * 40)
    
    # Calculate metrics for predicted headings only
    predicted_headings = test_df[test_df['is_heading_pred'] == 1].copy()
    
    # After filtering
    filtered_predicted_headings = filtered_df[filtered_df['is_heading_pred'] == 1].copy()
    
    # True/False Positives analysis
    original_fp = predicted_headings[predicted_headings['is_heading'] == 0]  # Original false positives
    original_tp = predicted_headings[predicted_headings['is_heading'] == 1]  # Original true positives
    
    # After filtering - check which ones remain as predictions
    remaining_predictions = filtered_df[filtered_df['is_heading_pred'] == 1]
    filtered_fp = remaining_predictions[remaining_predictions['is_heading'] == 0]  # Remaining false positives
    filtered_tp = remaining_predictions[remaining_predictions['is_heading'] == 1]  # Remaining true positives
    
    logger.info(f"\n🔍 DEBUG INFO:")
    logger.info(f"   - Original predictions (is_heading_pred=1): {len(predicted_headings)}")
    logger.info(f"   - Original false positives (pred=1, truth=0): {len(original_fp)}")
    logger.info(f"   - Original true positives (pred=1, truth=1): {len(original_tp)}")
    logger.info(f"   - Remaining predictions after filtering: {len(remaining_predictions)}")
    logger.info(f"   - Remaining false positives: {len(filtered_fp)}")
    logger.info(f"   - Remaining true positives: {len(filtered_tp)}")
    
    # Show which false positives were in the original data
    logger.info(f"\n📋 ORIGINAL FALSE POSITIVES:")
    for _, row in original_fp.iterrows():
        logger.info(f"   - '{row['text']}' (confidence: {row['heading_confidence']:.3f})")
    
    # Show which ones were filtered out
    filtered_out_fp = predicted_headings[(predicted_headings['is_heading'] == 0) & 
                                        (~predicted_headings.index.isin(remaining_predictions.index))]
    logger.info(f"\n🚫 FILTERED OUT FALSE POSITIVES:")
    for _, row in filtered_out_fp.iterrows():
        filter_reasons = filtered_df.loc[row.name, 'filter_reasons'] if 'filter_reasons' in filtered_df.columns else 'N/A'
        filter_score = filtered_df.loc[row.name, 'filter_score'] if 'filter_score' in filtered_df.columns else 0
        logger.info(f"   - '{row['text']}' (Score: {filter_score:.3f}) - {filter_reasons}")
    
    # Calculate reduction in false positives
    fp_reduction = len(original_fp) - len(filtered_fp)
    fp_reduction_rate = (fp_reduction / len(original_fp) * 100) if len(original_fp) > 0 else 0
    
    # Calculate preservation of true positives
    tp_preserved = len(filtered_tp)
    tp_preservation_rate = (tp_preserved / len(original_tp) * 100) if len(original_tp) > 0 else 0
    
    logger.info(f"📈 False Positive Reduction:")
    logger.info(f"   - Original false positives: {len(original_fp)}")
    logger.info(f"   - Filtered false positives: {len(filtered_fp)}")
    logger.info(f"   - Reduction: {fp_reduction} ({fp_reduction_rate:.1f}%)")
    
    logger.info(f"\n📈 True Positive Preservation:")
    logger.info(f"   - Original true positives: {len(original_tp)}")
    logger.info(f"   - Preserved true positives: {tp_preserved}")
    logger.info(f"   - Preservation rate: {tp_preservation_rate:.1f}%")
    
    # Show specific examples that were filtered
    logger.info(f"\n🔍 EXAMPLES OF FILTERED FALSE POSITIVES:")
    filtered_out = test_df[(test_df['is_heading_pred'] == 1) & 
                          (filtered_df['is_heading_pred'] == 0) & 
                          (test_df['is_heading'] == 0)]
    
    for _, row in filtered_out.iterrows():
        filter_reasons = filtered_df.loc[row.name, 'filter_reasons'] if 'filter_reasons' in filtered_df.columns else 'N/A'
        logger.info(f"   ❌ '{row['text']}' (Score: {filtered_df.loc[row.name, 'filter_score']:.3f}) - {filter_reasons}")
    
    # Show examples that were preserved
    logger.info(f"\n✅ EXAMPLES OF PRESERVED TRUE POSITIVES:")
    preserved = test_df[(test_df['is_heading_pred'] == 1) & 
                       (filtered_df['is_heading_pred'] == 1) & 
                       (test_df['is_heading'] == 1)]
    
    for _, row in preserved.head(5).iterrows():
        filter_reasons = filtered_df.loc[row.name, 'filter_reasons'] if 'filter_reasons' in filtered_df.columns else 'N/A'
        logger.info(f"   ✅ '{row['text']}' (Score: {filtered_df.loc[row.name, 'filter_score']:.3f}) - {filter_reasons}")
    
    # Show any incorrectly filtered true positives (should be minimal)
    incorrectly_filtered = test_df[(test_df['is_heading_pred'] == 1) & 
                                  (filtered_df['is_heading_pred'] == 0) & 
                                  (test_df['is_heading'] == 1)]
    
    if len(incorrectly_filtered) > 0:
        logger.warning(f"\n⚠️  INCORRECTLY FILTERED TRUE POSITIVES: {len(incorrectly_filtered)}")
        for _, row in incorrectly_filtered.iterrows():
            filter_reasons = filtered_df.loc[row.name, 'filter_reasons'] if 'filter_reasons' in filtered_df.columns else 'N/A'
            logger.warning(f"   ⚠️  '{row['text']}' (Score: {filtered_df.loc[row.name, 'filter_score']:.3f}) - {filter_reasons}")
    
    # Generate detailed filtering report
    logger.info(f"\n📋 Generating detailed filtering report...")
    report = filter_system.generate_filtering_report(filtered_df)
    
    logger.info(f"\n🎯 FILTERING PERFORMANCE SUMMARY:")
    logger.info(f"   📉 False Positive Reduction: {fp_reduction_rate:.1f}%")
    logger.info(f"   📈 True Positive Preservation: {tp_preservation_rate:.1f}%")
    
    overall_score = (fp_reduction_rate * 0.7) + (tp_preservation_rate * 0.3)
    logger.info(f"   🏆 Overall Performance Score: {overall_score:.1f}%")
    
    return filtered_df, report

def main():
    """Main function"""
    print("🧪 ENHANCED INTELLIGENT FILTERING TEST")
    print("=" * 50)
    print("🎯 Testing with specific false positive examples from requirements")
    print()
    
    try:
        filtered_df, report = test_filtering_performance()
        
        print("\n✅ Test completed successfully!")
        print("Check the logs above for detailed results.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        logging.error(f"Test error: {e}", exc_info=True)

if __name__ == "__main__":
    main()
