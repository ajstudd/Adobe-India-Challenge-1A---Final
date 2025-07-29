#!/usr/bin/env python3
"""
Hierarchical Numbering Analyzer for Heading Detection
====================================================

This module implements a comprehensive numbering system that recognizes and validates
hierarchical numbering patterns in headings, including decimal, roman numeral, and 
alphabetical systems as described by the user.

Numbering Systems Supported:
1. Decimal System: 1, 1.1, 1.1.1, 1.1.1.1, etc.
2. Roman Numeral System: I, I.I, I.I.I, etc. (using roman fundamentals)
3. Alphabetical System: A, A.1, A.1.a, etc.

The analyzer uses mathematical decimal system logic to understand hierarchical 
relationships and cycling patterns.

Author: AI Assistant
Date: July 29, 2025
"""

import re
import logging
import re
import logging
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class NumberingPattern:
    """Represents a detected numbering pattern"""
    pattern_type: str  # 'decimal', 'roman', 'alphabetical', 'mixed'
    level: int         # Hierarchical level (1, 2, 3, etc.)
    components: List[str]  # Individual components like ['1', '2'] for '1.2'
    raw_text: str      # Original text
    is_valid: bool     # Whether the pattern is valid
    next_expected: Optional[str] = None  # What we expect next at this level

class HierarchicalNumberingAnalyzer:
    """Analyze and validate hierarchical numbering patterns in headings"""
    
    def __init__(self):
        # Roman numeral fundamentals as requested
        self.roman_fundamentals = {
            'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000
        }
        
        # Roman numeral building blocks for proper validation
        self.valid_roman_patterns = [
            'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X',
            'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI', 'XVII', 'XVIII', 'XIX', 'XX',
            'XXI', 'XXII', 'XXIII', 'XXIV', 'XXV', 'XXVI', 'XXVII', 'XXVIII', 'XXIX', 'XXX',
            'XL', 'L', 'LX', 'LXX', 'LXXX', 'XC', 'C'
        ]
        
        # Alphabet fundamentals (A-Z, a-z)
        self.uppercase_letters = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
        self.lowercase_letters = [chr(i) for i in range(ord('a'), ord('z') + 1)]
        
        # Digits 0-9 as requested
        self.digits = [str(i) for i in range(10)]
        
        # Compiled regex patterns for efficient matching
        self.patterns = self._compile_patterns()
        
        # State tracking for sequence validation
        self.numbering_context = {}
        
        logger.info("🔢 Hierarchical Numbering Analyzer initialized!")
        logger.info(f"📊 Roman fundamentals: {list(self.roman_fundamentals.keys())}")
        logger.info(f"🔤 Alphabet range: {self.uppercase_letters[:5]}...{self.uppercase_letters[-5:]}")
        logger.info(f"🔢 Digit range: {self.digits}")
    
    def _compile_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for different numbering systems"""
        patterns = {}
        
        # Decimal system patterns (1, 1.1, 1.1.1, etc.)
        patterns['decimal_simple'] = re.compile(r'^\s*(\d+)\s*\.?\s*$')
        patterns['decimal_dotted'] = re.compile(r'^\s*(\d+)\.(\d+)(?:\.(\d+))?(?:\.(\d+))?\s*\.?\s*$')
        patterns['decimal_with_text'] = re.compile(r'^\s*(\d+(?:\.\d+)*)\.\s+(.+)$')
        patterns['decimal_multi_level'] = re.compile(r'^\s*(\d+(?:\.\d+){1,5})\s*\.?\s*(.*)$')
        
        # Roman numeral patterns (I, I.I, II.III, etc.)
        roman_pattern = r'[IVXLCDMivxlcdm]+'
        patterns['roman_simple'] = re.compile(f'^\\s*({roman_pattern})\\s*\\.?\\s*$', re.IGNORECASE)
        patterns['roman_dotted'] = re.compile(f'^\\s*({roman_pattern})\\.({roman_pattern})(?:\\.({roman_pattern}))?\\s*\\.?\\s*$', re.IGNORECASE)
        patterns['roman_with_text'] = re.compile(f'^\\s*({roman_pattern})\\.\\s+(.+)$', re.IGNORECASE)
        patterns['roman_standalone_with_text'] = re.compile(f'^\\s*({roman_pattern})\\s+(.+)$', re.IGNORECASE)
        
        # Alphabetical patterns (A, A.1, A.1.a, etc.)
        patterns['alpha_simple'] = re.compile(r'^\s*([A-Za-z])\s*\.?\s*$')
        patterns['alpha_dotted'] = re.compile(r'^\s*([A-Za-z])\.(\d+|[A-Za-z])(?:\.(\d+|[A-Za-z]))?\s*\.?\s*$')
        patterns['alpha_with_text'] = re.compile(r'^\s*([A-Za-z])\.\s+(.+)$')
        patterns['alpha_standalone_with_text'] = re.compile(r'^\s*([A-Za-z])\s+(.+)$')
        
        # Mixed patterns (A.1, 1.A, etc.)
        patterns['mixed_alpha_num'] = re.compile(r'^\s*([A-Za-z])\.(\d+)\s*\.?\s*$')
        patterns['mixed_num_alpha'] = re.compile(r'^\s*(\d+)\.([A-Za-z])\s*\.?\s*$')
        
        # Advanced patterns for complex hierarchies
        patterns['complex_hierarchy'] = re.compile(r'^\s*((?:\d+|[A-Za-z]|[IVXLCDMivxlcdm]+)(?:\.(?:\d+|[A-Za-z]|[IVXLCDMivxlcdm]+))*)\s*\.?\s*(.*)$', re.IGNORECASE)
        
        return patterns
    
    def roman_to_int(self, roman: str) -> int:
        """Convert roman numeral to integer using the fundamentals"""
        roman = roman.upper()
        total = 0
        prev_value = 0
        
        for char in reversed(roman):
            value = self.roman_fundamentals.get(char, 0)
            if value < prev_value:
                total -= value
            else:
                total += value
            prev_value = value
        
        return total
    
    def int_to_roman(self, num: int) -> str:
        """Convert integer to roman numeral"""
        values = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
        literals = ['M', 'CM', 'D', 'CD', 'C', 'XC', 'L', 'XL', 'X', 'IX', 'V', 'IV', 'I']
        
        result = ''
        for i, value in enumerate(values):
            count = num // value
            if count:
                result += literals[i] * count
                num -= value * count
        return result
    
    def is_valid_roman_numeral(self, roman: str) -> bool:
        """Check if a string is a valid roman numeral using fundamentals"""
        roman = roman.upper()
        
        # Check if all characters are valid roman numeral characters
        if not all(char in self.roman_fundamentals for char in roman):
            return False
        
        # Check against known valid patterns for common cases
        if roman in self.valid_roman_patterns:
            return True
        
        # For longer numerals, try conversion and back-conversion
        try:
            num = self.roman_to_int(roman)
            return self.int_to_roman(num) == roman and 1 <= num <= 3999
        except:
            return False
    
    def analyze_numbering_pattern(self, text: str) -> NumberingPattern:
        """Analyze text to detect and validate numbering patterns"""
        text = text.strip()
        
        # Try each pattern type
        for pattern_name, pattern in self.patterns.items():
            match = pattern.match(text)
            if match:
                return self._process_match(pattern_name, match, text)
        
        # No pattern found
        return NumberingPattern(
            pattern_type='none',
            level=0,
            components=[],
            raw_text=text,
            is_valid=False
        )
    
    def _process_match(self, pattern_name: str, match: re.Match, text: str) -> NumberingPattern:
        """Process a regex match to create a NumberingPattern"""
        groups = match.groups()
        
        if pattern_name.startswith('decimal'):
            return self._process_decimal_pattern(groups, text)
        elif pattern_name.startswith('roman'):
            return self._process_roman_pattern(groups, text, pattern_name)
        elif pattern_name.startswith('alpha'):
            return self._process_alpha_pattern(groups, text, pattern_name)
        elif pattern_name.startswith('mixed'):
            return self._process_mixed_pattern(groups, text)
        elif pattern_name == 'complex_hierarchy':
            return self._process_complex_pattern(groups, text)
        
        return NumberingPattern('unknown', 0, [], text, False)
    
    def _process_decimal_pattern(self, groups: Tuple, text: str) -> NumberingPattern:
        """Process decimal numbering patterns (1, 1.1, 1.1.1, etc.)"""
        if 'decimal_multi_level' in text or len(groups) > 1:
            # Multi-level: 1.1.1
            numbering_part = groups[0]
            components = numbering_part.split('.')
            level = len(components)
            
            # Validate each component is a valid number
            try:
                numeric_components = [int(comp) for comp in components]
                is_valid = all(comp > 0 for comp in numeric_components)
            except ValueError:
                is_valid = False
        else:
            # Simple: 1 or 1.
            try:
                components = [groups[0]]
                level = 1
                is_valid = int(groups[0]) > 0
            except (ValueError, IndexError):
                components = []
                level = 0
                is_valid = False
        
        return NumberingPattern(
            pattern_type='decimal',
            level=level,
            components=components,
            raw_text=text,
            is_valid=is_valid
        )
    
    def _process_roman_pattern(self, groups: Tuple, text: str, pattern_name: str = '') -> NumberingPattern:
        """Process roman numeral patterns (I, I.I, II.III, etc.)"""
        components = []
        level = 0
        is_valid = True
        
        if 'with_text' in pattern_name or 'standalone_with_text' in pattern_name:
            # Pattern like "I. Introduction" or "I Introduction"
            roman_part = groups[0]
            text_part = groups[1] if len(groups) > 1 else ''
            
            if self.is_valid_roman_numeral(roman_part):
                components = [roman_part.upper()]
                level = 1
            else:
                is_valid = False
        else:
            # Multiple roman numeral components
            for group in groups:
                if group:
                    components.append(group.upper())
                    level += 1
                    # Validate each roman numeral component
                    if not self.is_valid_roman_numeral(group):
                        is_valid = False
        
        return NumberingPattern(
            pattern_type='roman',
            level=level,
            components=components,
            raw_text=text,
            is_valid=is_valid
        )
    
    def _process_alpha_pattern(self, groups: Tuple, text: str, pattern_name: str = '') -> NumberingPattern:
        """Process alphabetical patterns (A, A.1, A.1.a, etc.)"""
        components = []
        level = 0
        is_valid = True
        
        if 'with_text' in pattern_name or 'standalone_with_text' in pattern_name:
            # Pattern like "A. Introduction" or "A Introduction" 
            alpha_part = groups[0]
            text_part = groups[1] if len(groups) > 1 else ''
            
            if len(alpha_part) == 1 and alpha_part.isalpha():
                components = [alpha_part]
                level = 1
            else:
                is_valid = False
        else:
            # Multiple components
            for group in groups:
                if group:
                    components.append(group)
                    level += 1
                    # Validate alphabetical components
                    if len(group) == 1 and group.isalpha():
                        continue  # Valid single letter
                    elif group.isdigit():
                        continue  # Valid number in mixed context
                    else:
                        is_valid = False
        
        return NumberingPattern(
            pattern_type='alphabetical',
            level=level,
            components=components,
            raw_text=text,
            is_valid=is_valid
        )
    
    def _process_mixed_pattern(self, groups: Tuple, text: str) -> NumberingPattern:
        """Process mixed patterns (A.1, 1.A, etc.)"""
        components = list(groups)
        level = len([g for g in groups if g])
        
        # Validate mixed components
        is_valid = True
        for comp in components:
            if comp and not (comp.isdigit() or (len(comp) == 1 and comp.isalpha())):
                is_valid = False
                break
        
        return NumberingPattern(
            pattern_type='mixed',
            level=level,
            components=components,
            raw_text=text,
            is_valid=is_valid
        )
    
    def _process_complex_pattern(self, groups: Tuple, text: str) -> NumberingPattern:
        """Process complex hierarchical patterns"""
        if not groups or not groups[0]:
            return NumberingPattern('none', 0, [], text, False)
        
        numbering_part = groups[0]
        components = numbering_part.split('.')
        level = len(components)
        
        # Determine pattern type based on components
        pattern_types = set()
        is_valid = True
        
        for comp in components:
            if comp.isdigit():
                pattern_types.add('decimal')
            elif self.is_valid_roman_numeral(comp):
                pattern_types.add('roman')
            elif len(comp) == 1 and comp.isalpha():
                pattern_types.add('alphabetical')
            else:
                is_valid = False
        
        # Determine overall pattern type
        if len(pattern_types) == 1:
            pattern_type = list(pattern_types)[0]
        elif len(pattern_types) > 1:
            pattern_type = 'mixed'
        else:
            pattern_type = 'unknown'
            is_valid = False
        
        return NumberingPattern(
            pattern_type=pattern_type,
            level=level,
            components=components,
            raw_text=text,
            is_valid=is_valid
        )
    
    def validate_sequence_consistency(self, patterns: List[NumberingPattern]) -> Dict[str, bool]:
        """Validate that a sequence of numbering patterns is logically consistent"""
        results = {
            'is_consistent': True,
            'has_proper_hierarchy': True,
            'has_sequential_numbering': True,
            'mixed_systems_valid': True
        }
        
        if not patterns:
            return results
        
        # Check hierarchy consistency
        prev_level = 0
        for pattern in patterns:
            if pattern.level > prev_level + 1:
                results['has_proper_hierarchy'] = False
            prev_level = pattern.level
        
        # Check sequential numbering within each level
        level_counters = {}
        for pattern in patterns:
            if pattern.pattern_type == 'decimal' and pattern.components:
                try:
                    for i, comp in enumerate(pattern.components):
                        level_key = f"decimal_{i}"
                        expected = level_counters.get(level_key, 0) + 1
                        actual = int(comp)
                        
                        if actual != expected and level_key not in level_counters:
                            level_counters[level_key] = actual
                        elif actual != expected:
                            results['has_sequential_numbering'] = False
                        else:
                            level_counters[level_key] = actual
                except ValueError:
                    results['has_sequential_numbering'] = False
        
        # Overall consistency
        results['is_consistent'] = all([
            results['has_proper_hierarchy'],
            results['has_sequential_numbering'],
            results['mixed_systems_valid']
        ])
        
        return results
    
    def generate_enhanced_patterns(self) -> Dict[str, re.Pattern]:
        """Generate enhanced regex patterns for the intelligent filter"""
        enhanced_patterns = {}
        
        # Enhanced decimal patterns
        enhanced_patterns['decimal_headings'] = re.compile(
            r'^\s*\d+(?:\.\d+)*\s*\.?\s*(?:[A-Z].*)?$'
        )
        
        # Enhanced roman numeral patterns - COMPREHENSIVE
        enhanced_patterns['roman_headings'] = re.compile(
            r'^\s*(?:[IVXLCDMivxlcdm]+(?:\.[IVXLCDMivxlcdm]+)*)\s*\.?\s*(?:[A-Z].*)?$'
        )
        
        # Enhanced alphabetical patterns
        enhanced_patterns['alphabetical_headings'] = re.compile(
            r'^\s*[A-Za-z](?:\.\w+)*\s*\.?\s*(?:[A-Z].*)?$'
        )
        
        # Mixed system patterns
        enhanced_patterns['mixed_headings'] = re.compile(
            r'^\s*(?:\d+\.[A-Za-z]|[A-Za-z]\.\d+)(?:\.\w+)*\s*\.?\s*(?:[A-Z].*)?$'
        )
        
        # Complex hierarchical patterns
        enhanced_patterns['complex_headings'] = re.compile(
            r'^\s*(?:\d+|[A-Za-z]|[IVXLCDMivxlcdm]+)(?:\.(?:\d+|[A-Za-z]|[IVXLCDMivxlcdm]+))*\s*\.?\s*(?:[A-Z].*)?$'
        )
        
        return enhanced_patterns
    
    def is_valid_heading_number(self, text: str) -> Tuple[bool, str, int]:
        """
        Determine if text contains a valid heading numbering pattern
        
        Returns:
            (is_valid, pattern_type, hierarchy_level)
        """
        pattern = self.analyze_numbering_pattern(text)
        
        # Additional validation: must be a proper heading pattern, not just any text
        if pattern.is_valid and pattern.level > 0:
            # For alphabetical patterns, require proper heading format (letter + dot + space + text OR letter + dot + end)
            if pattern.pattern_type == 'alphabetical':
                # Must be either "A." or "A. Something" format, not "Apple" or "Bcrypt.js"
                text_clean = text.strip()
                if not (re.match(r'^\s*[A-Za-z]\s*\.?\s*$', text_clean) or 
                       re.match(r'^\s*[A-Za-z]\.\s+[A-Z]', text_clean) or
                       re.match(r'^\s*[A-Za-z]\s+[A-Z]', text_clean)):
                    return False, 'none', 0
            
            # For roman patterns, require proper format too
            elif pattern.pattern_type == 'roman':
                text_clean = text.strip()
                # Must be proper roman numeral format, not just any word starting with roman letters
                if not (re.match(r'^\s*[IVXLCDMivxlcdm]+\s*\.?\s*$', text_clean) or 
                       re.match(r'^\s*[IVXLCDMivxlcdm]+\.\s+[A-Z]', text_clean) or
                       re.match(r'^\s*[IVXLCDMivxlcdm]+\s+[A-Z]', text_clean)):
                    return False, 'none', 0
                # Additional check: must be a valid roman numeral, not just any combination
                roman_part = re.match(r'^\s*([IVXLCDMivxlcdm]+)', text_clean).group(1)
                if not self.is_valid_roman_numeral(roman_part):
                    return False, 'none', 0
            
            return True, pattern.pattern_type, pattern.level
        
        return False, 'none', 0


def main():
    """Test the hierarchical numbering analyzer"""
    print("🔢 HIERARCHICAL NUMBERING ANALYZER")
    print("=" * 50)
    
    analyzer = HierarchicalNumberingAnalyzer()
    
    # Test cases covering all requested systems
    test_cases = [
        # Decimal system
        "1. Introduction",
        "1.1 Overview", 
        "1.1.1 Background",
        "1.1.1.1 Details",
        "2. Methodology",
        "2.1 Approach",
        
        # Roman numeral system
        "I. Introduction",
        "I.I Overview",
        "I.I.I Background", 
        "II. Methodology",
        "III. Results",
        "IV. Discussion",
        "V. Conclusion",
        
        # Alphabetical system
        "A. Introduction",
        "A.1 Overview",
        "A.1.a Background",
        "B. Methodology", 
        "C. Results",
        
        # Mixed systems
        "A.1 Mixed Example",
        "1.A Another Mixed",
        
        # Invalid cases
        "Just text",
        "123.456.789.012.345.678", # Too many levels
        "XYZ. Invalid Roman",
        
        # Edge cases from your JSON
        "Problem Analysis",
        "A. Product Definition", 
        "B. Feasibility Analysis",
        "C. Continuous Integration and Deployment",
    ]
    
    print("\n📊 TESTING NUMBERING PATTERNS:")
    print("=" * 40)
    
    for test_text in test_cases:
        pattern = analyzer.analyze_numbering_pattern(test_text)
        is_valid, pattern_type, level = analyzer.is_valid_heading_number(test_text)
        
        print(f"\nText: '{test_text}'")
        print(f"  Pattern Type: {pattern.pattern_type}")
        print(f"  Level: {pattern.level}")
        print(f"  Components: {pattern.components}")
        print(f"  Valid: {pattern.is_valid}")
        print(f"  Heading Valid: {is_valid} (Type: {pattern_type}, Level: {level})")
    
    print("\n🔧 ENHANCED PATTERNS FOR FILTER:")
    enhanced = analyzer.generate_enhanced_patterns()
    for name, pattern in enhanced.items():
        print(f"  {name}: {pattern.pattern}")


if __name__ == "__main__":
    main()
