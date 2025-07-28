#!/usr/bin/env python3
"""
Simple test for hierarchical numbering functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hierarchical_numbering_analyzer import HierarchicalNumberingAnalyzer
import pandas as pd

def test_direct_numbering():
    """Test the hierarchical numbering analyzer directly"""
    print("🔢 DIRECT HIERARCHICAL NUMBERING TEST")
    print("=" * 50)
    
    analyzer = HierarchicalNumberingAnalyzer()
    
    # Test cases from your JSON output that are being blocked
    test_cases = [
        "I. Introduction",
        "II. Methodology", 
        "A. Product Definition",
        "B. Feasibility Analysis",
        "C. Continuous Integration and Deployment",
        "VIII. Outcomes",
        "C.1 Docker-Centric Pipeline",
        "C.2 Jenkins Integration and Security Enforcement",
        "C.3 Frontend Deployment Workflow",
        "D. Deployment Strategy",
        "E. Oracle VPS-Based Deployment Strategy",
        "1. Overview",
        "1.1 Background",
        "1.1.1 Details",
        "A.1 Mixed Example",
        # Known false positives that should be blocked
        ", 2024. Available:",
        "Bcrypt.js",
        "Docker-based",
        "Initially,",
        "Problem Analysis"  # This might be legitimate
    ]
    
    print("\n📊 HIERARCHICAL NUMBERING ANALYSIS:")
    print("=" * 40)
    
    for text in test_cases:
        is_valid, pattern_type, level = analyzer.is_valid_heading_number(text)
        pattern = analyzer.analyze_numbering_pattern(text)
        
        status = "✅ VALID" if is_valid else "❌ INVALID"
        print(f"{status} | '{text:<45}' | Type: {pattern_type:<12} | Level: {level} | Components: {pattern.components}")

def test_simple_filter():
    """Test a simplified version of the filter logic"""
    print("\n🧠 SIMPLE FILTER TEST")
    print("=" * 50)
    
    try:
        from intelligent_filter import IntelligentFilter
        filter_system = IntelligentFilter()
        
        # Simple test cases
        test_texts = [
            "I. Introduction",
            "A. Product Definition", 
            "1. Overview",
            "C.1 Docker-Centric Pipeline",
            ", 2024. Available:",
            "Bcrypt.js"
        ]
        
        print("\nTesting individual text patterns:")
        for text in test_texts:
            # Test hierarchical numbering
            if filter_system.numbering_analyzer:
                is_valid, pattern_type, level = filter_system.numbering_analyzer.is_valid_heading_number(text)
                print(f"Hierarchical: '{text}' -> Valid: {is_valid}, Type: {pattern_type}, Level: {level}")
            
            # Test positive patterns
            is_positive, positive_pattern = filter_system.check_positive_patterns(text)
            print(f"Positive: '{text}' -> Match: {is_positive}, Pattern: {positive_pattern}")
            
            # Test exclusion patterns
            is_excluded, exclusion_reason = filter_system.check_exclusion_patterns(text)
            print(f"Exclusion: '{text}' -> Excluded: {is_excluded}, Reason: {exclusion_reason}")
            print("-" * 60)
            
    except Exception as e:
        print(f"Error testing filter: {e}")

if __name__ == "__main__":
    test_direct_numbering()
    test_simple_filter()
