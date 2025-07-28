#!/usr/bin/env python3
"""
Enhanced Intelligent Filtering Test Suite
Tests the complete enhanced filtering system with metadata analysis
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
import sys

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

from intelligent_filter import IntelligentFilter
from enhanced_metadata_extractor import EnhancedMetadataExtractor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_enhanced_test_data():
    """Create comprehensive test data covering various scenarios"""
    test_data = [
        # TRUE HEADINGS - Should be preserved with proper levels
        {'text': 'Introduction', 'font_size': 16, 'is_heading_pred': 1, 'bold': True, 'page': 1, 'y0': 100, 'y1': 115, 'x0': 50, 'x1': 200, 'line_position_on_page': 1, 'expected': True, 'expected_level': 'H1', 'description': 'Document introduction'},
        {'text': 'Project Overview', 'font_size': 18, 'is_heading_pred': 1, 'bold': True, 'page': 1, 'y0': 50, 'y1': 68, 'x0': 50, 'x1': 250, 'line_position_on_page': 2, 'expected': True, 'expected_level': 'H1', 'description': 'Main section heading'},
        {'text': 'Literature Review', 'font_size': 14, 'is_heading_pred': 1, 'bold': True, 'page': 2, 'y0': 750, 'y1': 765, 'x0': 50, 'x1': 220, 'line_position_on_page': 3, 'expected': True, 'expected_level': 'H2', 'description': 'Section heading'},
        {'text': 'Methodology', 'font_size': 14, 'is_heading_pred': 1, 'bold': True, 'page': 3, 'y0': 700, 'y1': 715, 'x0': 50, 'x1': 180, 'line_position_on_page': 4, 'expected': True, 'expected_level': 'H2', 'description': 'Section heading'},
        {'text': 'Data Collection Procedures', 'font_size': 12, 'is_heading_pred': 1, 'bold': True, 'page': 3, 'y0': 600, 'y1': 612, 'x0': 50, 'x1': 300, 'line_position_on_page': 5, 'expected': True, 'expected_level': 'H3', 'description': 'Subsection heading'},
        {'text': 'Results and Analysis', 'font_size': 14, 'is_heading_pred': 1, 'bold': True, 'page': 4, 'y0': 750, 'y1': 765, 'x0': 50, 'x1': 240, 'line_position_on_page': 6, 'expected': True, 'expected_level': 'H2', 'description': 'Section heading'},
        {'text': 'Statistical Analysis', 'font_size': 12, 'is_heading_pred': 1, 'bold': True, 'page': 4, 'y0': 650, 'y1': 662, 'x0': 50, 'x1': 220, 'line_position_on_page': 7, 'expected': True, 'expected_level': 'H3', 'description': 'Subsection heading'},
        {'text': 'Conclusion', 'font_size': 16, 'is_heading_pred': 1, 'bold': True, 'page': 5, 'y0': 750, 'y1': 768, 'x0': 50, 'x1': 180, 'line_position_on_page': 8, 'expected': True, 'expected_level': 'H1', 'description': 'Major section'},
        
        # FALSE POSITIVES - Should be filtered out
        {'text': 'Junaid Ahmad', 'font_size': 12, 'is_heading_pred': 1, 'bold': False, 'page': 1, 'y0': 200, 'y1': 212, 'x0': 50, 'x1': 150, 'line_position_on_page': 9, 'expected': False, 'expected_level': None, 'description': 'Personal name'},
        {'text': 'Registration : 12315906', 'font_size': 10, 'is_heading_pred': 1, 'bold': False, 'page': 1, 'y0': 180, 'y1': 190, 'x0': 50, 'x1': 200, 'line_position_on_page': 10, 'expected': False, 'expected_level': None, 'description': 'Registration number'},
        {'text': 'Bcrypt.js', 'font_size': 11, 'is_heading_pred': 1, 'bold': False, 'page': 2, 'y0': 400, 'y1': 411, 'x0': 50, 'x1': 120, 'line_position_on_page': 11, 'expected': False, 'expected_level': None, 'description': 'Technical library name'},
        {'text': 'docker-based', 'font_size': 10, 'is_heading_pred': 1, 'bold': False, 'page': 2, 'y0': 350, 'y1': 360, 'x0': 50, 'x1': 140, 'line_position_on_page': 12, 'expected': False, 'expected_level': None, 'description': 'Technical term'},
        {'text': 'the system architecture was designed', 'font_size': 12, 'is_heading_pred': 1, 'bold': False, 'page': 3, 'y0': 500, 'y1': 512, 'x0': 50, 'x1': 350, 'line_position_on_page': 13, 'expected': False, 'expected_level': None, 'description': 'Sentence fragment starting with "the"'},
        {'text': 'this approach provides better performance,', 'font_size': 12, 'is_heading_pred': 1, 'bold': False, 'page': 3, 'y0': 450, 'y1': 462, 'x0': 50, 'x1': 380, 'line_position_on_page': 14, 'expected': False, 'expected_level': None, 'description': 'Sentence ending with comma'},
        {'text': 'Docker', 'font_size': 10, 'is_heading_pred': 1, 'bold': False, 'page': 2, 'y0': 320, 'y1': 330, 'x0': 50, 'x1': 100, 'line_position_on_page': 15, 'expected': False, 'expected_level': None, 'description': 'Single word technical term'},
        {'text': 'initially,', 'font_size': 11, 'is_heading_pred': 1, 'bold': False, 'page': 2, 'y0': 300, 'y1': 311, 'x0': 50, 'x1': 110, 'line_position_on_page': 16, 'expected': False, 'expected_level': None, 'description': 'Adverb with comma'},
        {'text': '• Item one', 'font_size': 11, 'is_heading_pred': 1, 'bold': False, 'page': 2, 'y0': 280, 'y1': 291, 'x0': 50, 'x1': 130, 'line_position_on_page': 17, 'expected': False, 'expected_level': None, 'description': 'Bullet point'},
        {'text': 'https://example.com', 'font_size': 10, 'is_heading_pred': 1, 'bold': False, 'page': 2, 'y0': 260, 'y1': 270, 'x0': 50, 'x1': 180, 'line_position_on_page': 18, 'expected': False, 'expected_level': None, 'description': 'URL'},
        {'text': 'Table 1: Results of the experiment showing various parameters and their corresponding values', 'font_size': 12, 'is_heading_pred': 1, 'bold': False, 'page': 4, 'y0': 400, 'y1': 412, 'x0': 50, 'x1': 600, 'line_position_on_page': 19, 'expected': False, 'expected_level': None, 'description': 'Very long table caption'},
        
        # EDGE CASES
        {'text': 'API', 'font_size': 14, 'is_heading_pred': 1, 'bold': True, 'page': 2, 'y0': 600, 'y1': 615, 'x0': 50, 'x1': 80, 'line_position_on_page': 20, 'expected': True, 'expected_level': 'H3', 'description': 'Short technical heading with formatting'},
        {'text': 'Machine Learning Algorithms', 'font_size': 13, 'is_heading_pred': 1, 'bold': True, 'page': 3, 'y0': 550, 'y1': 563, 'x0': 50, 'x1': 280, 'line_position_on_page': 21, 'expected': True, 'expected_level': 'H3', 'description': 'Technical heading'},
        {'text': 'ABSTRACT', 'font_size': 14, 'is_heading_pred': 1, 'bold': True, 'page': 1, 'y0': 600, 'y1': 615, 'x0': 50, 'x1': 140, 'line_position_on_page': 22, 'expected': True, 'expected_level': 'H2', 'description': 'All caps section'},
        {'text': 'References', 'font_size': 14, 'is_heading_pred': 1, 'bold': True, 'page': 6, 'y0': 750, 'y1': 765, 'x0': 50, 'x1': 160, 'line_position_on_page': 23, 'expected': True, 'expected_level': 'H1', 'description': 'References section'},
    ]
    
    return pd.DataFrame(test_data)

def run_enhanced_filtering_test():
    """Run comprehensive test of enhanced intelligent filtering"""
    logger.info("🧪 Starting Enhanced Intelligent Filtering Test Suite")
    
    # Create test data
    test_df = create_enhanced_test_data()
    logger.info(f"📊 Created test dataset with {len(test_df)} samples")
    logger.info(f"   • True headings: {test_df['expected'].sum()}")
    logger.info(f"   • False positives: {(~test_df['expected']).sum()}")
    
    # Add confidence scores (simulating ML model output)
    np.random.seed(42)  # For reproducible results
    test_df['heading_confidence'] = np.random.uniform(0.6, 0.95, len(test_df))
    
    # Boost confidence for true headings
    test_df.loc[test_df['expected'] == True, 'heading_confidence'] = np.random.uniform(0.8, 0.95, test_df['expected'].sum())
    
    # Lower confidence for some false positives
    false_positive_mask = test_df['expected'] == False
    test_df.loc[false_positive_mask, 'heading_confidence'] = np.random.uniform(0.6, 0.85, false_positive_mask.sum())
    
    # Initialize enhanced intelligent filter with metadata extractor
    try:
        metadata_extractor = EnhancedMetadataExtractor()
        intelligent_filter = IntelligentFilter(metadata_extractor=metadata_extractor)
        logger.info("✅ Enhanced intelligent filter initialized with metadata extractor")
    except Exception as e:
        logger.warning(f"⚠️  Falling back to basic filter: {e}")
        intelligent_filter = IntelligentFilter()
    
    # Apply enhanced filtering
    logger.info("🔍 Applying enhanced intelligent filtering...")
    filtered_df = intelligent_filter.apply_intelligent_filtering(test_df.copy())
    
    # Analyze results
    logger.info("\n📈 ENHANCED FILTERING RESULTS ANALYSIS")
    logger.info("=" * 60)
    
    # Overall statistics
    original_predictions = test_df['is_heading_pred'].sum()
    final_predictions = filtered_df['is_heading_pred'].sum()
    reduction_rate = ((original_predictions - final_predictions) / original_predictions) * 100
    
    logger.info(f"📊 Overall Statistics:")
    logger.info(f"   • Original ML predictions: {original_predictions}")
    logger.info(f"   • Final predictions after filtering: {final_predictions}")
    logger.info(f"   • Reduction rate: {reduction_rate:.1f}%")
    
    # Heading level distribution
    if 'final_heading_level' in filtered_df.columns:
        heading_levels = filtered_df[filtered_df['is_heading_pred'] == 1]['final_heading_level'].value_counts()
        logger.info(f"   • Heading level distribution: {dict(heading_levels)}")
    
    # Accuracy analysis
    true_headings = test_df['expected'] == True
    false_positives = test_df['expected'] == False
    
    # True positive preservation
    preserved_true_headings = filtered_df.loc[true_headings, 'is_heading_pred'].sum()
    total_true_headings = true_headings.sum()
    true_positive_rate = (preserved_true_headings / total_true_headings) * 100
    
    # False positive reduction
    remaining_false_positives = filtered_df.loc[false_positives, 'is_heading_pred'].sum()
    total_false_positives = false_positives.sum()
    false_positive_reduction = ((total_false_positives - remaining_false_positives) / total_false_positives) * 100
    
    logger.info(f"\n🎯 Accuracy Metrics:")
    logger.info(f"   • True positive preservation: {preserved_true_headings}/{total_true_headings} ({true_positive_rate:.1f}%)")
    logger.info(f"   • False positive reduction: {total_false_positives - remaining_false_positives}/{total_false_positives} ({false_positive_reduction:.1f}%)")
    
    # Detailed analysis by category
    logger.info(f"\n🔍 Detailed Analysis:")
    
    # Check preserved true headings
    preserved_mask = (test_df['expected'] == True) & (filtered_df['is_heading_pred'] == 1)
    if preserved_mask.any():
        logger.info(f"   ✅ Preserved True Headings ({preserved_mask.sum()}):")
        for idx, row in test_df[preserved_mask].iterrows():
            level = filtered_df.loc[idx, 'final_heading_level'] if 'final_heading_level' in filtered_df.columns else 'N/A'
            logger.info(f"      • \"{row['text'][:50]}...\" → {level} | {row['description']}")
    
    # Check incorrectly filtered true headings
    lost_mask = (test_df['expected'] == True) & (filtered_df['is_heading_pred'] == 0)
    if lost_mask.any():
        logger.info(f"   ❌ Lost True Headings ({lost_mask.sum()}):")
        for idx, row in test_df[lost_mask].iterrows():
            reason = filtered_df.loc[idx, 'filter_reasons'] if 'filter_reasons' in filtered_df.columns else 'N/A'
            logger.info(f"      • \"{row['text'][:50]}...\" | {row['description']} | Reason: {reason}")
    
    # Check successfully filtered false positives
    filtered_fp_mask = (test_df['expected'] == False) & (filtered_df['is_heading_pred'] == 0)
    if filtered_fp_mask.any():
        logger.info(f"   ✅ Successfully Filtered False Positives ({filtered_fp_mask.sum()}):")
        for idx, row in test_df[filtered_fp_mask].iterrows():
            reason = filtered_df.loc[idx, 'filter_reasons'] if 'filter_reasons' in filtered_df.columns else 'N/A'
            logger.info(f"      • \"{row['text'][:50]}...\" | {row['description']}")
    
    # Check remaining false positives
    remaining_fp_mask = (test_df['expected'] == False) & (filtered_df['is_heading_pred'] == 1)
    if remaining_fp_mask.any():
        logger.info(f"   ⚠️  Remaining False Positives ({remaining_fp_mask.sum()}):")
        for idx, row in test_df[remaining_fp_mask].iterrows():
            level = filtered_df.loc[idx, 'final_heading_level'] if 'final_heading_level' in filtered_df.columns else 'N/A'
            reason = filtered_df.loc[idx, 'filter_reasons'] if 'filter_reasons' in filtered_df.columns else 'N/A'
            logger.info(f"      • \"{row['text'][:50]}...\" → {level} | {row['description']} | Reason: {reason}")
    
    # Overall performance score
    if total_true_headings > 0 and total_false_positives > 0:
        performance_score = (true_positive_rate * 0.7) + (false_positive_reduction * 0.3)
        logger.info(f"\n🏆 Overall Performance Score: {performance_score:.1f}/100")
        
        if performance_score >= 90:
            logger.info("   🌟 EXCELLENT - System ready for production!")
        elif performance_score >= 80:
            logger.info("   ✅ GOOD - Minor improvements needed")
        elif performance_score >= 70:
            logger.info("   ⚠️  FAIR - Significant improvements needed")
        else:
            logger.info("   ❌ POOR - Major rework required")
    
    return filtered_df

def main():
    """Main test function"""
    try:
        # Run the enhanced filtering test
        results_df = run_enhanced_filtering_test()
        
        # Save detailed results
        output_path = Path("enhanced_filtering_test_results.csv")
        results_df.to_csv(output_path, index=False)
        logger.info(f"\n💾 Detailed results saved to: {output_path}")
        
        logger.info("\n🎉 Enhanced Intelligent Filtering Test Complete!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
