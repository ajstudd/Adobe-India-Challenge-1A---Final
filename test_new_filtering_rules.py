#!/usr/bin/env python3
"""
Test script for new filtering rules
Test the enhanced filtering rules requested by the user:
1. Heading starting with special chars like &, _, comma, ., \\, %,*,!, ~ should be discarded
2. $ can be there but just next to it should be a number or letter
3. Heading starting with small case should be discarded
4. Use parts of speech or grammar rules to discard single words false heading like For, the, them, their
5. Heading should be more than 3 characters
"""

import sys
import os
sys.path.append('.')

from intelligent_filter import IntelligentFilter
import pandas as pd

def test_new_filtering_rules():
    """Test the new filtering rules"""
    print("🧪 TESTING NEW FILTERING RULES")
    print("=" * 50)
    
    # Initialize the filter
    filter_system = IntelligentFilter()
    
    # Test cases for each new rule
    test_cases = [
        # Rule 1: Bad special characters at start (should be rejected)
        ("&Introduction", True, "starts_with_bad_special_char"),
        ("_Section", True, "starts_with_bad_special_char"),
        (",Overview", True, "starts_with_bad_special_char"),
        (".Chapter", True, "starts_with_bad_special_char"),
        ("\\Background", True, "starts_with_bad_special_char"),
        ("%Analysis", True, "starts_with_bad_special_char"),
        ("*Results", True, "starts_with_bad_special_char"),
        ("!Conclusion", True, "starts_with_bad_special_char"),
        ("~Summary", True, "starts_with_bad_special_char"),
        
        # Rule 2: $ rule ($ must be followed by number or letter)
        ("$100", False, "valid_dollar_with_number"),  # Should be accepted
        ("$Project", False, "valid_dollar_with_letter"),  # Should be accepted
        ("$ invalid", True, "dollar_not_followed_by_alphanumeric"),  # Should be rejected
        ("$-test", True, "dollar_not_followed_by_alphanumeric"),  # Should be rejected
        
        # Rule 3: Lowercase start (should be rejected)
        ("introduction", True, "starts_with_lowercase"),  # Should be rejected
        ("background", True, "starts_with_lowercase"),  # Should be rejected
        ("iPhone", False, "technical_exception"),  # Exception - should be accepted
        
        # Rule 4: Function words (should be rejected)
        ("for", True, "single_word_function_word"),
        ("the", True, "single_word_function_word"),  
        ("them", True, "single_word_function_word"),
        ("their", True, "single_word_function_word"),
        ("these", True, "single_word_function_word"),
        ("and", True, "single_word_function_word"),
        ("but", True, "single_word_function_word"),
        ("with", True, "single_word_function_word"),
        ("however", True, "single_word_function_word"),
        ("therefore", True, "single_word_function_word"),
        
        # Rule 5: Length rule (should be more than 3 characters)
        ("abc", True, "too_short_less_than_4_chars"),
        ("ab", True, "too_short_less_than_4_chars"),
        ("a", False, "exception_single_letter"),  # Exception for single letters
        ("I", False, "exception_roman_numeral"),  # Exception for roman numerals
        ("1", False, "exception_number"),  # Exception for numbers
        ("test", False, "valid_length"),  # Should be accepted
        
        # Valid headings that should NOT be rejected
        ("Introduction", False, "valid_heading"),
        ("1. Overview", False, "numbered_heading"),
        ("I. Background", False, "roman_numeral_heading"),
        ("A. Analysis", False, "letter_heading"),
        ("Chapter 1", False, "chapter_heading"),
        ("Summary", False, "legitimate_single_word"),
        ("Results", False, "legitimate_single_word"),
        ("Methodology", False, "legitimate_single_word"),
        ("1.1 Details", False, "hierarchical_numbering"),
        ("II. Findings", False, "roman_numeral_with_text"),
        ("B.1 Subsection", False, "mixed_numbering"),
        
        # Edge cases
        ("", True, "empty_string"),
        ("   ", True, "whitespace_only"),
        ("A", False, "single_capital_letter"),
        ("1", False, "single_number"),
        ("X", False, "single_roman_numeral"),
    ]
    
    print("\n📊 Testing each rule:")
    print("=" * 40)
    
    passed = 0
    failed = 0
    
    for text, should_reject, test_description in test_cases:
        # Create a dummy row for testing
        row = pd.Series({
            'text': text,
            'font_size': 12,
            'page': 1,
            'line_position_on_page': 0.5
        })
        
        # Create empty context
        context = {}
        
        # Test the filtering
        actually_rejected, reasons = filter_system.apply_rule_based_filters(text, row, context)
        
        # Check if the result matches expectation
        test_passed = actually_rejected == should_reject
        
        if test_passed:
            passed += 1
            status = "✅ PASS"
        else:
            failed += 1
            status = "❌ FAIL"
        
        print(f"{status} | '{text}' | Expected: {'REJECT' if should_reject else 'ACCEPT'} | "
              f"Actual: {'REJECT' if actually_rejected else 'ACCEPT'} | {test_description}")
        
        if not test_passed:
            print(f"     Reasons: {reasons}")
    
    print("\n" + "=" * 50)
    print(f"📈 RESULTS: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    print("=" * 50)
    
    # Test exclusion patterns as well
    print("\n🔍 Testing exclusion patterns:")
    print("=" * 40)
    
    exclusion_test_cases = [
        ("&bad", True, "bad_special_char"),
        ("$", True, "dollar_rule"),  # Changed from "$invalid" to just "$"
        ("for", True, "function_word"),
        ("Introduction", False, "legitimate_heading"),
        ("1. Section", False, "numbered_section"),
    ]
    
    for text, should_exclude, description in exclusion_test_cases:
        excluded, pattern = filter_system.check_exclusion_patterns(text)
        test_passed = excluded == should_exclude
        status = "✅ PASS" if test_passed else "❌ FAIL"
        print(f"{status} | '{text}' | Expected: {'EXCLUDE' if should_exclude else 'INCLUDE'} | "
              f"Actual: {'EXCLUDE' if excluded else 'INCLUDE'} | Pattern: {pattern}")
    
    return passed, failed

def test_with_current_json_problems():
    """Test with problematic entries from the current JSON output"""
    print("\n🎯 TESTING WITH CURRENT JSON PROBLEMS")
    print("=" * 50)
    
    filter_system = IntelligentFilter()
    
    # These are examples from the current JSON that should be filtered out
    problem_cases = [
        "These",      # Too short and function word
        "The",        # Function word  
        "analysis",   # Starts with lowercase
        "results",    # Starts with lowercase
        "Other",      # Could be function word
        "Percent",    # Could be legitimate but in wrong context
    ]
    
    print("Testing problematic entries that should be rejected:")
    for text in problem_cases:
        row = pd.Series({
            'text': text,
            'font_size': 12,
            'page': 34,
            'line_position_on_page': 0.5
        })
        
        rejected, reasons = filter_system.apply_rule_based_filters(text, row, {})
        status = "✅ FILTERED" if rejected else "❌ NOT FILTERED"
        print(f"{status} | '{text}' | Reasons: {reasons}")

if __name__ == "__main__":
    passed, failed = test_new_filtering_rules()
    test_with_current_json_problems()
    
    if failed == 0:
        print("\n🎉 All tests passed! New filtering rules are working correctly.")
    else:
        print(f"\n⚠️  {failed} tests failed. Please review the implementation.")
