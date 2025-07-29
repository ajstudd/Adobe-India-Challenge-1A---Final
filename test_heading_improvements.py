#!/usr/bin/env python3
"""
Test Script for Heading Detection Improvements
==============================================

This script tests the improvements made to:
1. Fix exclusion of numbered headings like "1.Introduction" and "I.Introduction"
2. Improve heading level detection with metadata-based logic
3. Verify proper heading hierarchy assignment (H1/H2/H3)

Author: AI Assistant
Date: July 29, 2025
"""

import sys
import os
import pandas as pd
import re
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the improved systems
try:
    from intelligent_filter import IntelligentFilter
    INTELLIGENT_FILTER_AVAILABLE = True
except ImportError as e:
    INTELLIGENT_FILTER_AVAILABLE = False
    logger.warning(f"⚠️ Intelligent filter not available: {e}")

try:
    from enhanced_metadata_extractor import EnhancedMetadataExtractor  
    ENHANCED_METADATA_AVAILABLE = True
except ImportError as e:
    ENHANCED_METADATA_AVAILABLE = False
    logger.warning(f"⚠️ Enhanced metadata extractor not available: {e}")

try:
    from generate_json_output import JSONOutputGenerator
    JSON_GENERATOR_AVAILABLE = True
except ImportError as e:
    JSON_GENERATOR_AVAILABLE = False
    logger.warning(f"⚠️ JSON generator not available: {e}")

def test_heading_patterns():
    """Test if numbered headings are properly recognized"""
    print("🧪 TESTING HEADING PATTERN RECOGNITION")
    print("=" * 50)
    
    # Test cases that should be recognized as headings
    test_headings = [
        "1.Introduction",
        "I.Introduction", 
        "II.Methodology",
        "1. Introduction",
        "I. Introduction",
        "1.1 Overview",
        "2.3 System Architecture",
        "A. Problem Analysis",
        "B. Solution Design",
        "CHAPTER 1",
        "SECTION 2",
        "1 Introduction",  # Space instead of dot
        "2 System Design",
        "1.1.1 Detailed Analysis",
        "(a) Sub-component",
        "a) Implementation"
    ]
    
    # Test cases that should NOT be recognized as headings
    test_non_headings = [
        "1. this is a sentence",  # lowercase after number
        "http://www.example.com",
        "This is a regular paragraph.",
        "bcrypt.js",
        "Initially,",
        "Docker-based",
        "The system employs several technologies.",
        "page 1",
        "Registration: 123456"
    ]
    
    if not INTELLIGENT_FILTER_AVAILABLE:
        print("❌ Cannot test - Intelligent filter not available")
        return
    
    filter_system = IntelligentFilter()
    
    print("\n📋 Test Results for VALID HEADINGS:")
    print("-" * 40)
    
    for text in test_headings:
        # Test exclusion patterns
        is_excluded, exclusion_reason = filter_system.check_exclusion_patterns(text)
        
        # Test positive patterns
        is_positive, positive_reason = filter_system.check_positive_patterns(text)
        
        status = "✅ PASS" if not is_excluded or is_positive else "❌ FAIL"
        print(f"{status} '{text}'")
        if is_excluded and not is_positive:
            print(f"     Excluded by: {exclusion_reason}")
        if is_positive:
            print(f"     Matched: {positive_reason}")
    
    print("\n📋 Test Results for NON-HEADINGS:")
    print("-" * 40)
    
    for text in test_non_headings:
        # Test exclusion patterns
        is_excluded, exclusion_reason = filter_system.check_exclusion_patterns(text)
        
        # Test positive patterns
        is_positive, positive_reason = filter_system.check_positive_patterns(text)
        
        status = "✅ PASS" if is_excluded and not is_positive else "❌ FAIL"
        print(f"{status} '{text}'")
        if is_excluded:
            print(f"     Correctly excluded by: {exclusion_reason}")
        if is_positive:
            print(f"     Incorrectly matched: {positive_reason}")

def test_heading_level_detection():
    """Test metadata-based heading level detection"""
    print("\n\n🏗️ TESTING HEADING LEVEL DETECTION")
    print("=" * 50)
    
    if not JSON_GENERATOR_AVAILABLE:
        print("❌ Cannot test - JSON generator not available")
        return
    
    # Create test data
    test_data = {
        'text': [
            'CHAPTER 1: INTRODUCTION',     # Should be H1
            '1.Introduction',              # Should be H1  
            'I.Introduction',              # Should be H1
            '1.1 Overview',                # Should be H2
            '2.3 System Design',           # Should be H2
            'A. Problem Analysis',         # Should be H2
            '1.1.1 Implementation',        # Should be H3
            'Methodology',                 # Should be H2/H3 based on font
            'References'                   # Should be H1
        ],
        'font_size': [18, 16, 16, 14, 14, 14, 12, 13, 15],
        'page': [1, 1, 1, 1, 2, 2, 2, 3, 3],
        'line_position_on_page': [1, 5, 8, 12, 3, 7, 15, 5, 20],
        'y0': [750, 700, 650, 600, 720, 650, 500, 680, 400]
    }
    
    df = pd.DataFrame(test_data)
    
    # Test with JSON generator
    generator = JSONOutputGenerator()
    
    # Calculate font percentiles for the dataset
    font_percentiles = {}
    for p in [60, 70, 75, 80, 85, 90, 95, 98]:
        font_percentiles[p] = df['font_size'].quantile(p/100)
    
    print("\n📊 Font Size Analysis:")
    print(f"Font sizes in test data: {sorted(df['font_size'].tolist())}")
    for p in [75, 90, 95]:
        print(f"{p}th percentile: {font_percentiles[p]:.1f}")
    
    print("\n📋 Heading Level Test Results:")
    print("-" * 40)
    
    for idx, row in df.iterrows():
        predicted_level = generator.determine_heading_level(row, font_percentiles)
        print(f"'{row['text']}' -> {predicted_level} (font: {row['font_size']})")

def test_enhanced_metadata_extraction():
    """Test enhanced metadata extraction"""
    print("\n\n🔍 TESTING ENHANCED METADATA EXTRACTION")
    print("=" * 50)
    
    if not ENHANCED_METADATA_AVAILABLE:
        print("❌ Cannot test - Enhanced metadata extractor not available")
        return
    
    # Create test data with problematic headings
    test_data = {
        'text': [
            '1.Introduction',              # Should be detected and assigned H1
            'I.Introduction',              # Should be detected and assigned H1
            '1.1 System Overview',         # Should be detected and assigned H2
            'II.Methodology',              # Should be detected and assigned H1
            '2.3 Implementation Details',  # Should be detected and assigned H2
            'bcrypt.js',                   # Should be filtered out
            'Initially,',                  # Should be filtered out
            'A. Problem Analysis',         # Should be detected and assigned H2
            'References',                  # Should be detected as H1
            'The system employs'           # Should be filtered out
        ],
        'font_size': [16, 16, 14, 16, 14, 10, 12, 14, 15, 11],
        'page': [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
        'line_position_on_page': [1, 5, 8, 1, 5, 8, 12, 1, 15, 20],
        'x0': [50, 50, 70, 50, 70, 60, 80, 50, 50, 50],
        'y0': [750, 700, 650, 720, 650, 600, 550, 700, 400, 350],
        'x1': [200, 200, 220, 200, 250, 120, 140, 180, 150, 180],
        'y1': [770, 720, 670, 740, 670, 615, 565, 720, 415, 365]
    }
    
    df = pd.DataFrame(test_data)
    
    # Initialize extractor
    extractor = EnhancedMetadataExtractor()
    
    # Extract metadata
    enhanced_df = extractor.extract_comprehensive_metadata(df)
    
    print("\n📋 Enhanced Metadata Results:")
    print("-" * 40)
    
    # Display key results
    for idx, row in enhanced_df.iterrows():
        text = row['text']
        likelihood = row.get('heading_likelihood', 0)
        recommended_level = row.get('recommended_level', 'N/A')
        structure_score = row.get('structure_score', 0)
        
        print(f"'{text}':")
        print(f"  Likelihood: {likelihood:.3f}")
        print(f"  Recommended Level: {recommended_level}")
        print(f"  Structure Score: {structure_score}")
        print()

def main():
    """Run all tests"""
    print("🧪 HEADING DETECTION IMPROVEMENTS TEST SUITE")
    print("=" * 60)
    print("Testing fixes for:")
    print("  1. Numbered headings like '1.Introduction' and 'I.Introduction'")
    print("  2. Metadata-based heading level detection (H1/H2/H3)")
    print("  3. Enhanced pattern recognition and filtering")
    print()
    
    try:
        # Test 1: Pattern recognition
        test_heading_patterns()
        
        # Test 2: Level detection
        test_heading_level_detection()
        
        # Test 3: Enhanced metadata
        test_enhanced_metadata_extraction()
        
        print("\n\n✅ All tests completed!")
        print("Check the results above to verify improvements are working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
