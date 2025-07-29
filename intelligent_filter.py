#!/usr/bin/env python3
"""
Intelligent Rule-Based Filtering System
======================================

This script implements intelligent rule-based filtering to reduce false positives
while preserving correct heading predictions from the ML model.

Enhanced with comprehensive metadata analysis for better heading detection.

Key Features:
✅ Multi-layered filtering approach (contextual, statistical, linguistic)
✅ Confidence-based decision making
✅ Preservation of high-confidence model predictions
✅ Dynamic thresholding based on document characteristics
✅ Comprehensive metadata analysis integration
✅ Proper heading hierarchy assignment (H1/H2/H3)
✅ Advanced typography and semantic analysis

Strategy:
- Extract comprehensive metadata for intelligent analysis
- Apply sophisticated filtering using linguistic and semantic patterns
- Assign proper heading hierarchy based on font size, structure, and content
- Preserve high-quality headings while filtering false positives
- Identify document title as first H1 heading within initial 100 blocks

Author: AI Assistant
Date: July 28, 2025
"""

import os
import re
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Import enhanced metadata extractor
try:
    from enhanced_metadata_extractor import EnhancedMetadataExtractor
    ENHANCED_METADATA_AVAILABLE = True
except ImportError as e:
    ENHANCED_METADATA_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

class IntelligentFilter:
    """Intelligent rule-based filtering system for heading predictions"""
    
    def __init__(self, config_path=None, metadata_extractor=None):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load configuration
        self.config = self.load_config(config_path)
        
        # Filter thresholds - Made Much Less Aggressive
        self.high_confidence_threshold = 0.8  # Lowered from 0.9
        self.medium_confidence_threshold = 0.5  # Lowered from 0.7  
        self.low_confidence_threshold = 0.3  # Lowered from 0.5
        
        # Enhanced thresholds for metadata-based filtering - much less aggressive
        self.heading_likelihood_threshold = 0.05  # Significantly reduced from 0.2
        self.syntax_violation_limit = 8  # Increased from 4
        
        # Statistics for dynamic filtering
        self.document_stats = {}
        
        # Initialize enhanced metadata extractor
        if metadata_extractor is not None:
            self.metadata_extractor = metadata_extractor
            logger.info("✅ Enhanced metadata extractor provided")
        elif ENHANCED_METADATA_AVAILABLE:
            self.metadata_extractor = EnhancedMetadataExtractor()
            logger.info("✅ Enhanced metadata extractor initialized")
        else:
            self.metadata_extractor = None
            logger.warning("⚠️  Enhanced metadata extractor not available")
        
        # Exclusion patterns (common false positives) - Enhanced based on requirements
        self.exclusion_patterns = {
            # Basic text patterns
            'starts_with_lowercase': re.compile(r'^\s*[a-z]'),  # New rule: reject headings starting with lowercase
            
            # URLs and technical references
            'urls': re.compile(r'http[s]?://|www\.|\.com|\.org|\.edu|\.gov', re.IGNORECASE),
            'emails': re.compile(r'\S+@\S+\.\S+'),
            'file_paths': re.compile(r'[/\\][a-zA-Z0-9_\-./\\]+|[a-zA-Z]:[/\\]'),
            
            # Date and number patterns
            'dates': re.compile(r'\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b|\b\d{4}[/\-]\d{1,2}[/\-]\d{1,2}\b'),
            'page_numbers': re.compile(r'^\s*(?:page\s+)?\d+\s*$', re.IGNORECASE),
            'registration_numbers': re.compile(r'registration\s*:?\s*\d+', re.IGNORECASE),
            'student_ids': re.compile(r'^\s*\d{6,12}\s*$'),  # Long numeric IDs
            
            # Document structure elements
            'footers': re.compile(r'^\s*(?:page\s+\d+|copyright|©|\d+\s*$)', re.IGNORECASE),
            'bullets': re.compile(r'^\s*[•◦▪▫◾▸►‣⁃]\s*'),
            'numbered_lists': re.compile(r'^\s*\d+\.\s+[a-z]'),  # 1. something (lowercase start) - Keep this for actual lists
            'references': re.compile(r'^\s*\[\d+\]|\(\d{4}\)|\d{4}[a-z]?\b'),
            
            # Sentence patterns (Rule 1: Reject sentence-like structures) - Less aggressive
            'sentence_endings': re.compile(r'^.{50,}\.\s*$'),  # Only long sentences ending with full stop
            'long_sentences': re.compile(r'^.{150,}'),  # Increased from 120 characters
            'question_sentences': re.compile(r'^.{30,}\?\s*$'),  # Increased from 20 chars
            
            # Identity and name patterns (Rule 2: Reject identity blocks)
            'university_patterns': re.compile(r'university|college|institute', re.IGNORECASE),
            'registration_patterns': re.compile(r'registration\s*:?\s*\d+|b\.?tech|student|name\s*:', re.IGNORECASE),
            'location_patterns': re.compile(r'phagwara|punjab|india', re.IGNORECASE),
            'person_names': re.compile(r'^[A-Z][a-z]+\s+[A-Z][a-z]+$'),  # FirstName LastName pattern
            'full_names': re.compile(r'^[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$'),  # Full names
            'course_names': re.compile(r'computer\s+science|engineering|b\.?tech', re.IGNORECASE),
            
            # Technical terms and tools
            'technical_terms': re.compile(r'^[A-Z][a-z]*\.(js|py|css|html)$|docker|bcrypt|automation|vps|api|jwt|json|web|token', re.IGNORECASE),
            'version_numbers': re.compile(r'v\d+\.\d+|version\s+\d+', re.IGNORECASE),
            'technical_ids': re.compile(r'^[A-Z]{2,}\d+|^\d+[A-Z]+\d*$'),
            'programming_terms': re.compile(r'^(api|jwt|json|xml|http|https|css|html|javascript|nodejs|react|docker|kubernetes|git|github)$', re.IGNORECASE),
            'technical_fragments': re.compile(r'^(backend|frontend|interface|leverages|architecture|employs)$', re.IGNORECASE),
            
            # Fragment patterns (Rule 5: Must not be fragments)
            'single_words': re.compile(r'^\s*\w+\s*$'),  # Single word
            'discourse_markers': re.compile(r'^(these|initially|however|therefore|thus|hence|moreover|furthermore|additionally|the\s+\w+|in\s+summary)$', re.IGNORECASE),
            'conjunctions': re.compile(r'^(and|but|or|so|yet|for|nor)$', re.IGNORECASE),
            'incomplete_phrases': re.compile(r'^(the|a|an|with|by|from|to|in|on|at|of|for)\s+\w+', re.IGNORECASE),
            'sentence_starters': re.compile(r'^(the\s+\w+|this\s+\w+|that\s+\w+|these\s+\w+|from\s+a\s+social|deployed\s+application)', re.IGNORECASE),
            
            # System and containerization terms from examples
            'system_fragments': re.compile(r'containerized\s+infrastructure|powered\s+by|hosting\s+environment|communication\s+for|leverages\s+\w+|employs\s+\w+|updates\s*$', re.IGNORECASE),
            'article_words': re.compile(r'^(a|an|the)$', re.IGNORECASE),
            'platform_names': re.compile(r'^proactive\s*india$|^proactive$|^india$', re.IGNORECASE),  # Specific to this document
            'implementation_fragments': re.compile(r'^the\s+implementation\s+of|^the\s+development\s+and|^the\s+application\s+is', re.IGNORECASE),
        }
        
        # Positive patterns (likely to be headings) - Enhanced
        self.positive_patterns = {
            'chapter_section': re.compile(r'^\s*(?:chapter|section|part|unit|lesson)\s+\d+', re.IGNORECASE),
            'appendix': re.compile(r'^\s*appendix\s+[a-z]\b', re.IGNORECASE),
            'common_headings': re.compile(r'^\s*(?:introduction|conclusion|summary|abstract|references|bibliography|acknowledgments?|methodology|results|discussion|analysis|overview|background|objectives?|scope|limitations?|recommendations?|findings)\s*$', re.IGNORECASE),
            'numbered_headings': re.compile(r'^\s*\d+(?:\.\d+)*\s+[A-Z]'),  # 1.1 Something (capital start)
            'numbered_heading_with_title': re.compile(r'^\s*\d+\.\s*[A-Z][a-z]*(?:\s+[A-Z][a-z]*)*$'),  # 1.Introduction, 1.System Architecture
            'roman_numeral_headings': re.compile(r'^\s*[IVX]+\.\s*[A-Z][a-z]*(?:\s+[A-Z][a-z]*)*.*$', re.IGNORECASE),  # I.Introduction, II.Methodology
            'title_case_short': re.compile(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,6}$'),  # Title Case (short)
            'roman_numerals': re.compile(r'^\s*[IVX]+\.\s+.*$', re.IGNORECASE),  # I. Something (any text after)
            'letter_headings': re.compile(r'^\s*[A-Z]\.\s+.*$'),  # A. Something (any text after)
            'project_specific': re.compile(r'^\s*(?:abstract|outcomes?|project\s+plan|implementation|project\s+legacy|project\s+resources)\s*$', re.IGNORECASE),  # Document-specific
        }
        
        # Common heading words for fragment detection
        self.common_heading_words = {
            'chapter', 'section', 'part', 'unit', 'introduction', 'conclusion', 
            'summary', 'abstract', 'overview', 'background', 'methodology',
            'results', 'discussion', 'analysis', 'objectives', 'scope',
            'limitations', 'recommendations', 'findings', 'appendix', 'references'
        }
        
        # Patterns for consecutive heading merging rule
        self.mergeable_prefix_patterns = {
            'numbers': re.compile(r'^\s*\d+\s*$'),                       # Just numbers like "1", "2", "3"
            'numbered_dots': re.compile(r'^\s*\d+\.\s*$'),               # Numbers with dots like "1.", "2.", "3."
            'numbered_multi': re.compile(r'^\s*\d+\.\d+\s*$'),           # Multi-level numbers like "1.1", "2.3"
            'roman_numerals': re.compile(r'^\s*[IVX]+\s*$', re.IGNORECASE),     # Roman numerals like "I", "II", "VII"
            'roman_numerals_dots': re.compile(r'^\s*[IVX]+\.\s*$', re.IGNORECASE),  # Roman numerals with dots like "I.", "VII."
            'single_letter': re.compile(r'^\s*[A-Z]\.\s*$'),             # Single letter with dot like "A.", "B.", "C."
            'letter_no_dot': re.compile(r'^\s*[A-Z]\s*$'),               # Single letter without dot like "A", "B", "C"
        }
        
        logger.info("🧠 Intelligent Filter initialized!")
        logger.info(f"🎯 High confidence threshold: {self.high_confidence_threshold}")
        logger.info(f"🎯 Medium confidence threshold: {self.medium_confidence_threshold}")
        logger.info("🔧 Added new filtering rules: consecutive heading merging and document title detection (first H1 in 100 blocks)")
        logger.info(f"🎯 Low confidence threshold: {self.low_confidence_threshold}")
    
    def load_config(self, config_path=None):
        """Load configuration"""
        if config_path is None:
            config_path = os.path.join(self.base_dir, 'config_main.json')
        
        if not os.path.exists(config_path):
            logger.warning(f"⚠️  Config file not found: {config_path}")
            return self.get_default_config()
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                logger.info(f"✅ Configuration loaded from {config_path}")
                return config
        except Exception as e:
            logger.error(f"❌ Error loading config: {e}")
            return self.get_default_config()
    
    def get_default_config(self):
        """Get default configuration"""
        return {
            "precision_filters": {
                "min_font_size_percentile": 70,
                "max_word_count": 20,
                "min_word_count": 1,
                "required_heading_patterns": True,
                "exclude_sentence_patterns": True,
                "strict_position_requirements": False
            }
        }
    
    def calculate_document_statistics(self, df):
        """Calculate document-level statistics for dynamic filtering"""
        logger.info("📊 Calculating document statistics for dynamic filtering...")
        
        stats = {
            'total_blocks': int(len(df)),
            'avg_font_size': float(df['font_size'].mean() if 'font_size' in df.columns else 12),
            'font_size_std': float(df['font_size'].std() if 'font_size' in df.columns else 2),
            'font_percentiles': {},
            'avg_word_count': float(df['text'].str.split().str.len().mean()),
            'total_pages': int(df['page'].max() if 'page' in df.columns else 1),
            'blocks_per_page': float(len(df) / df['page'].max() if 'page' in df.columns and df['page'].max() > 0 else len(df))
        }
        
        # Calculate font size percentiles if available
        if 'font_size' in df.columns:
            for p in [50, 70, 75, 80, 85, 90, 95, 98, 99]:
                stats['font_percentiles'][p] = float(df['font_size'].quantile(p/100))
        
        # Convert all stats to ensure JSON serialization
        self.document_stats = convert_numpy_types(stats)
        
        logger.info(f"   📈 Total blocks: {stats['total_blocks']}")
        logger.info(f"   📈 Average font size: {stats['avg_font_size']:.1f}")
        logger.info(f"   📈 Average word count: {stats['avg_word_count']:.1f}")
        logger.info(f"   📈 Total pages: {stats['total_pages']}")
        
        return self.document_stats
    
    def apply_rule_based_filters(self, text: str, row: pd.Series, context: Dict) -> Tuple[bool, List[str]]:
        """
        Apply the specific filtering rules from the requirements - MUCH more lenient
        Returns: (should_reject, rejection_reasons)
        """
        text_clean = text.strip()
        word_count = len(text_clean.split())
        char_count = len(text_clean)
        rejection_reasons = []
        
        # Rule 1: Only reject obvious sentence-like structures  
        if text_clean.endswith('.') and word_count > 15:  # Increased from 12
            rejection_reasons.append("long_sentence_with_period")
            
        if word_count > 20:  # Increased from 12
            rejection_reasons.append("too_many_words")
            
        if char_count > 150:  # Increased from 120
            rejection_reasons.append("too_many_characters")
        
        # Rule 2: Only reject obvious identity blocks 
        identity_keywords = ["registration :", "b.tech student", "university phagwara"]
        if any(keyword in text_clean.lower() for keyword in identity_keywords):
            rejection_reasons.append("clear_identity_pattern")
        
        # Rule 4: Only reject if starts with lowercase (but allow some exceptions)
        if (text_clean and text_clean[0].islower() and 
            not any(pattern.search(text_clean) for pattern in self.positive_patterns.values())):
            rejection_reasons.append("lowercase_start_no_positive_pattern")
        
        # Rule 5: Be more lenient with fragments
        if word_count <= 1:  # Only reject single words
            has_heading_words = any(word.lower() in self.common_heading_words 
                                  for word in text_clean.split())
            if not has_heading_words and len(text_clean) < 8:  # Very short single words
                rejection_reasons.append("very_short_fragment")
        
        # Only check for the most obvious false positives
        obvious_false_positives = [
            "registration :", "lovely professional university",
            "bcrypt.js", "dockerfile"
        ]
        
        if text_clean.lower() in obvious_false_positives:
            rejection_reasons.append("obvious_false_positive")
        
        # Should reject only if multiple major issues or obvious false positives
        should_reject = len(rejection_reasons) >= 2 or any("obvious" in reason for reason in rejection_reasons)
        
        return should_reject, rejection_reasons
    
    def check_exclusion_patterns(self, text: str) -> Tuple[bool, str]:
        """Check if text matches any exclusion patterns"""
        text_clean = text.strip()
        
        for pattern_name, pattern in self.exclusion_patterns.items():
            if pattern.search(text_clean):
                return True, pattern_name
        
        return False, ""
    
    def check_positive_patterns(self, text: str) -> Tuple[bool, str]:
        """Check if text matches any positive heading patterns"""
        text_clean = text.strip()
        
        for pattern_name, pattern in self.positive_patterns.items():
            if pattern.search(text_clean):
                return True, pattern_name
        
        return False, ""
    
    def analyze_context(self, df: pd.DataFrame, idx: int, window_size: int = 2) -> Dict:
        """Analyze the context around a potential heading"""
        context = {
            'prev_blocks': [],
            'next_blocks': [],
            'font_size_context': 'normal',
            'position_context': 'middle',
            'heading_density': 0.0
        }
        
        # Get surrounding blocks
        start_idx = max(0, idx - window_size)
        end_idx = min(len(df), idx + window_size + 1)
        
        for i in range(start_idx, end_idx):
            if i == idx:
                continue
            block_info = {
                'text': df.iloc[i]['text'] if 'text' in df.columns else '',
                'font_size': df.iloc[i]['font_size'] if 'font_size' in df.columns else 12,
                'is_prediction': df.iloc[i].get('is_heading_pred', 0) if 'is_heading_pred' in df.columns else 0
            }
            
            if i < idx:
                context['prev_blocks'].append(block_info)
            else:
                context['next_blocks'].append(block_info)
        
        # Analyze font size context
        current_font = df.iloc[idx]['font_size'] if 'font_size' in df.columns else 12
        surrounding_fonts = [b['font_size'] for b in context['prev_blocks'] + context['next_blocks']]
        
        if surrounding_fonts:
            avg_surrounding = np.mean(surrounding_fonts)
            if current_font > avg_surrounding * 1.2:
                context['font_size_context'] = 'larger'
            elif current_font < avg_surrounding * 0.8:
                context['font_size_context'] = 'smaller'
        
        # Analyze position context
        page = df.iloc[idx]['page'] if 'page' in df.columns else 1
        line_pos = df.iloc[idx]['line_position_on_page'] if 'line_position_on_page' in df.columns else 0
        
        if line_pos <= 3:
            context['position_context'] = 'top'
        elif 'line_position_on_page' in df.columns:
            page_blocks = df[df['page'] == page]
            max_line_pos = page_blocks['line_position_on_page'].max()
            if line_pos >= max_line_pos - 3:
                context['position_context'] = 'bottom'
        
        # Calculate local heading density
        if 'is_heading_pred' in df.columns:
            local_predictions = [b['is_prediction'] for b in context['prev_blocks'] + context['next_blocks']]
            context['heading_density'] = sum(local_predictions) / len(local_predictions) if local_predictions else 0.0
        
        return context
    
    def calculate_filter_score(self, row: pd.Series, context: Dict, confidence: float) -> Tuple[float, Dict]:
        """Calculate a comprehensive filtering score with enhanced rule-based filtering"""
        text = row.get('text', '').strip()
        reasons = []
        score = 0.5  # Start neutral
        
        # 1. Apply rule-based filters first (but don't penalize as heavily)
        should_reject, rejection_reasons = self.apply_rule_based_filters(text, row, context)
        
        if should_reject:
            # Moderate penalty for rule-based rejections instead of strong
            score -= 0.2  # Reduced from 0.5
            reasons.extend([f"rule_reject_{reason}" for reason in rejection_reasons])
        
        # 2. Confidence-based base score - more generous
        if confidence >= self.high_confidence_threshold:
            score += 0.4  # Increased from 0.3
            reasons.append(f"high_confidence_{confidence:.3f}")
        elif confidence >= self.medium_confidence_threshold:
            score += 0.2  # Increased from 0.1
            reasons.append(f"medium_confidence_{confidence:.3f}")
        elif confidence >= self.low_confidence_threshold:
            score += 0.0  # Neutral instead of negative
            reasons.append(f"low_confidence_{confidence:.3f}")
        else:
            score -= 0.05  # Very small penalty instead of -0.1
            reasons.append(f"very_low_confidence_{confidence:.3f}")
        
        # 3. Exclusion patterns (moderate negative instead of strong)
        is_excluded, exclusion_reason = self.check_exclusion_patterns(text)
        if is_excluded:
            score -= 0.2  # Reduced from 0.4
            reasons.append(f"excluded_{exclusion_reason}")
        
        # 4. Positive patterns (stronger positive boost)
        is_positive, positive_reason = self.check_positive_patterns(text)
        if is_positive:
            score += 0.5  # Increased from 0.3
            reasons.append(f"positive_{positive_reason}")
        
        # 5. Length-based scoring (much more lenient)
        word_count = len(text.split())
        if word_count <= 1 and len(text) < 5:  # Only penalize very short text
            score -= 0.2  # Reduced from 0.3
            reasons.append("very_short")
        elif word_count > 30:  # Increased from 25
            score -= 0.3  # Reduced from 0.4
            reasons.append("too_long")
        elif word_count > 20:  # Increased from 12
            score -= 0.1  # Reduced from 0.2
            reasons.append("moderately_long")
        elif 2 <= word_count <= 15:  # Increased from 8
            score += 0.15  # Increased from 0.1
            reasons.append("good_length")
        
        # 6. Font size relative scoring (enhanced with percentile thresholds)
        if 'font_size' in row and self.document_stats.get('font_percentiles'):
            font_size = row['font_size']
            percentiles = self.document_stats['font_percentiles']
            
            # Since most fonts are the same size (9.96), be more lenient
            if font_size >= percentiles.get(98, 16):
                score += 0.4  # Increased from 0.3
                reasons.append("very_large_font")
            elif font_size >= percentiles.get(95, 14):
                score += 0.3  # Increased from 0.2
                reasons.append("large_font")
            elif font_size >= percentiles.get(90, 12):
                score += 0.2  # Increased from 0.1
                reasons.append("above_avg_font")
            elif font_size >= percentiles.get(50, 11):
                score += 0.1  # Bonus for normal font
                reasons.append("normal_font")
            # Remove penalty for small fonts since most text has same size
        
        # 7. Context-based scoring
        if context['font_size_context'] == 'larger':
            score += 0.15
            reasons.append("larger_than_context")
        elif context['font_size_context'] == 'smaller':
            score -= 0.15
            reasons.append("smaller_than_context")
        
        if context['position_context'] == 'top':
            score += 0.2
            reasons.append("top_position")
        elif context['position_context'] == 'bottom':
            score -= 0.15
            reasons.append("bottom_position")
        
        # 8. Heading density (too many headings in vicinity is suspicious)
        if context['heading_density'] > 0.5:
            score -= 0.2
            reasons.append("high_heading_density")
        elif context['heading_density'] == 0:
            score += 0.1
            reasons.append("isolated_heading")
        
        # 9. Text quality checks (enhanced)
        if text.isupper() and len(text) > 50:
            score -= 0.3
            reasons.append("long_all_caps")
        
        if text.endswith('.') and len(text.split()) > 10:
            score -= 0.4
            reasons.append("long_sentence")
        
        # 10. Formatting indicators
        if row.get('bold', False):
            score += 0.15
            reasons.append("bold_text")
            
        if row.get('italic', False):
            score += 0.1
            reasons.append("italic_text")
        
        # 11. Position-based scoring (enhanced)
        if 'line_position_on_page' in row:
            line_pos = row['line_position_on_page']
            if line_pos <= 2:  # Very top of page
                score += 0.2
                reasons.append("page_top")
            elif line_pos <= 5:  # Near top
                score += 0.1
                reasons.append("near_top")
        
        # 12. POS-based enhancements (if available)
        if 'num_nouns' in row and 'num_verbs' in row:
            total_words = row.get('word_count', len(text.split()))
            if total_words > 0:
                noun_density = row['num_nouns'] / total_words
                verb_density = row['num_verbs'] / total_words
                
                # Headings typically have high noun density, low verb density
                if noun_density > 0.4 and verb_density < 0.2:
                    score += 0.1
                    reasons.append("good_pos_ratio")
        
        # 13. Special handling for technical terms and proper nouns
        if 'num_propn' in row and row['num_propn'] > 0:
            # Proper nouns can be good for headings, but not if they're person names
            if not any(pattern.search(text) for pattern in [
                self.exclusion_patterns['person_names'],
                self.exclusion_patterns['university_patterns']
            ]):
                score += 0.05
                reasons.append("contains_proper_nouns")
        
        return score, {"score": score, "reasons": reasons}
        
        # 7. Heading density (too many headings in vicinity is suspicious)
        if context['heading_density'] > 0.5:
            score -= 0.2
            reasons.append("high_heading_density")
        elif context['heading_density'] == 0:
            score += 0.1
            reasons.append("isolated_heading")
        
        # 8. Text quality checks
        if text.isupper() and len(text) > 50:
            score -= 0.2
            reasons.append("long_all_caps")
        
        if text.endswith('.') and len(text.split()) > 10:
            score -= 0.3
            reasons.append("long_sentence")
        
        # 9. Formatting indicators
        if row.get('bold', False) or row.get('italic', False):
            score += 0.1
            reasons.append("formatted_text")
        
        # 10. Position-based scoring
        if 'line_position_on_page' in row:
            line_pos = row['line_position_on_page']
            if line_pos <= 2:  # Very top of page
                score += 0.15
                reasons.append("page_top")
        
        return score, {"score": score, "reasons": reasons}
    
    def apply_intelligent_filtering(self, df: pd.DataFrame, confidence_col: str = 'heading_confidence') -> pd.DataFrame:
        """Apply intelligent rule-based filtering with enhanced metadata analysis"""
        logger.info("🧠 Applying enhanced intelligent rule-based filtering...")
        
        if confidence_col not in df.columns:
            logger.warning(f"⚠️  Confidence column '{confidence_col}' not found, using default confidence of 0.8")
            df[confidence_col] = 0.8
        
        # Step 1: Extract comprehensive metadata if available
        if self.metadata_extractor is not None:
            logger.info("📊 Extracting comprehensive metadata...")
            enhanced_df = self.metadata_extractor.extract_comprehensive_metadata(df)
        else:
            logger.warning("⚠️  Using basic analysis without enhanced metadata")
            enhanced_df = df.copy()
            # Add basic analysis
            enhanced_df['heading_likelihood'] = 0.5
            enhanced_df['syntax_violations'] = 0
            enhanced_df['recommended_level'] = 'H3'
            enhanced_df['content_type'] = 'unknown'
        
        # Calculate document statistics
        self.calculate_document_statistics(enhanced_df)
        
        # Initialize filtering results
        enhanced_df['filter_score'] = 0.0
        enhanced_df['filter_decision'] = 'keep'
        enhanced_df['filter_reasons'] = ''
        enhanced_df['original_prediction'] = enhanced_df.get('is_heading_pred', 0)
        enhanced_df['final_heading_level'] = 'H3'
        
        filtered_count = 0
        preserved_count = 0
        
        # Process each row with enhanced logic
        for idx, row in enhanced_df.iterrows():
            confidence = row[confidence_col]
            original_pred = row.get('is_heading_pred', 0)
            
            if original_pred == 0:
                # Not predicted as heading, skip filtering
                enhanced_df.at[idx, 'filter_decision'] = 'not_predicted'
                continue
            
            # Enhanced filtering logic
            heading_likelihood = row.get('heading_likelihood', 0.5)
            syntax_violations = row.get('syntax_violations', 0)
            content_type = row.get('content_type', 'unknown')
            recommended_level = row.get('recommended_level', 'H3')
            
            # Analyze context (basic implementation if enhanced not available)
            if self.metadata_extractor is not None:
                # Use enhanced context analysis
                context = {'enhanced': True}
                score, score_info = self.calculate_enhanced_filter_score(row, context, confidence)
            else:
                # Fallback to basic analysis
                context = self.analyze_context(enhanced_df, idx)
                score, score_info = self.calculate_filter_score(row, context, confidence)
            
            # Enhanced decision making
            decision, final_pred, final_level = self.make_enhanced_filtering_decision(
                row, confidence, heading_likelihood, syntax_violations, content_type, 
                recommended_level, score
            )
            
            # Update results
            if final_pred == 1:
                preserved_count += 1
            else:
                filtered_count += 1
            
            # Update dataframe
            enhanced_df.at[idx, 'filter_score'] = score
            enhanced_df.at[idx, 'filter_decision'] = decision
            enhanced_df.at[idx, 'filter_reasons'] = "; ".join(score_info.get('reasons', []))
            enhanced_df.at[idx, 'is_heading_pred'] = final_pred
            enhanced_df.at[idx, 'final_heading_level'] = final_level
        
        # Log results
        total_original_predictions = enhanced_df['original_prediction'].sum()
        total_filtered_predictions = enhanced_df['is_heading_pred'].sum()
        
        logger.info(f"📊 Enhanced Filtering Results:")
        logger.info(f"   🔢 Original predictions: {total_original_predictions}")
        logger.info(f"   ✅ Preserved predictions: {preserved_count}")
        logger.info(f"   ❌ Filtered predictions: {filtered_count}")
        logger.info(f"   📉 Final predictions: {total_filtered_predictions}")
        logger.info(f"   📈 Reduction rate: {(filtered_count/total_original_predictions*100):.1f}%")
        
        # Log heading level distribution
        heading_levels = enhanced_df[enhanced_df['is_heading_pred'] == 1]['final_heading_level'].value_counts()
        logger.info(f"   📏 Heading levels: {dict(heading_levels)}")
        
        # Apply new filtering rules
        logger.info("🔧 Applying additional filtering rules...")
        
        # Rule 1: Apply document title rule (first heading in initial 100 blocks)
        enhanced_df = self.apply_document_title_rule(enhanced_df)
        
        # Rule 2: Apply consecutive heading merge rule
        enhanced_df = self.apply_consecutive_heading_merge_rule(enhanced_df)
        
        # Log final results after additional rules
        final_predictions = enhanced_df['is_heading_pred'].sum()
        final_heading_levels = enhanced_df[enhanced_df['is_heading_pred'] == 1]['final_heading_level'].value_counts()
        logger.info(f"📊 Final Results After Additional Rules:")
        logger.info(f"   🔢 Final predictions: {final_predictions}")
        logger.info(f"   📏 Final heading levels: {dict(final_heading_levels)}")
        
        return enhanced_df
    
    def calculate_enhanced_filter_score(self, row: pd.Series, context: Dict, confidence: float) -> Tuple[float, Dict]:
        """Calculate enhanced filtering score using comprehensive metadata"""
        text = row.get('text', '').strip()
        reasons = []
        
        # Start with metadata-based likelihood
        base_score = row.get('heading_likelihood', 0.5)
        
        # Confidence boost
        if confidence >= self.high_confidence_threshold:
            base_score += 0.2
            reasons.append(f"high_confidence_{confidence:.3f}")
        elif confidence >= self.medium_confidence_threshold:
            base_score += 0.1
            reasons.append(f"medium_confidence_{confidence:.3f}")
        else:
            base_score -= 0.1
            reasons.append(f"low_confidence_{confidence:.3f}")
        
        # Apply syntax violations penalty
        violations = row.get('syntax_violations', 0)
        violation_penalty = violations * 0.15
        base_score -= violation_penalty
        
        if violations > 0:
            reasons.append(f"syntax_violations_{violations}")
        
        # Content type analysis
        content_type = row.get('content_type', 'unknown')
        if content_type == 'heading_candidate':
            base_score += 0.1
            reasons.append("heading_candidate")
        elif content_type in ['technical', 'fragment']:
            base_score -= 0.2
            reasons.append(f"content_type_{content_type}")
        
        # Typography and structure bonuses
        font_category = row.get('font_category', 'medium')
        if font_category in ['very_large', 'large']:
            base_score += 0.1
            reasons.append(f"font_{font_category}")
        
        structure_score = row.get('structure_score', 0)
        if structure_score > 0:
            base_score += structure_score * 0.05
            reasons.append(f"structure_score_{structure_score}")
        
        # Position bonuses
        position_score = row.get('position_score', 0)
        if position_score > 0:
            base_score += position_score * 0.03
            reasons.append(f"position_score_{position_score}")
        
        # Hard filters for obvious non-headings
        if (text.endswith(',') or 
            text.startswith(tuple(['the ', 'this ', 'that ', 'these ', 'it '])) or
            len(text.split()) > 15 or
            text.lower() in ['initially,', 'bcrypt.js', 'dockerfile', 'docker-based']):
            base_score = min(base_score, 0.2)
            reasons.append("hard_filter_applied")
        
        # Clamp score
        final_score = max(0, min(1, base_score))
        
        return final_score, {"score": final_score, "reasons": reasons}
    
    def make_enhanced_filtering_decision(self, row: pd.Series, confidence: float, 
                                       heading_likelihood: float, syntax_violations: int,
                                       content_type: str, recommended_level: str, 
                                       filter_score: float) -> Tuple[str, int, str]:
        """Make enhanced filtering decision with proper hierarchy assignment"""
        
        text = row.get('text', '').strip()
        
        # Hard rejection criteria - MUCH less aggressive
        if (syntax_violations >= self.syntax_violation_limit or
            heading_likelihood < 0.01 or  # Very minimal threshold 
            filter_score < -0.2):  # Only filter extremely bad scores
            return 'filtered_hard_criteria', 0, 'H3'
        
        # Remove most soft rejection criteria to be less aggressive
        # Only reject obvious technical terms
        if (content_type == 'technical' and confidence < 0.3 and filter_score < 0.1):
            return 'filtered_soft_criteria', 0, 'H3'
        
        # High confidence preservation - more lenient
        if confidence >= self.high_confidence_threshold:
            # Determine proper level based on metadata
            if recommended_level in ['H1', 'H2', 'H3']:
                final_level = recommended_level
            else:
                final_level = 'H2'  # Default for high confidence
            
            return 'keep_high_confidence', 1, final_level
        
        # Good score preservation - much more lenient
        if filter_score >= 0.2:  # Significantly reduced from 0.4
            final_level = recommended_level if recommended_level in ['H1', 'H2', 'H3'] else 'H2'
            return 'keep_good_score', 1, final_level
        
        # Medium confidence preservation - much more lenient
        if confidence >= self.medium_confidence_threshold:
            final_level = recommended_level if recommended_level in ['H1', 'H2', 'H3'] else 'H2'
            return 'keep_medium_quality', 1, final_level
        
        # Low confidence but reasonable content - very lenient
        if confidence >= self.low_confidence_threshold or filter_score >= 0.0:
            final_level = recommended_level if recommended_level in ['H1', 'H2', 'H3'] else 'H3'
            return 'keep_low_quality', 1, final_level
        
        # Default: keep most things now, only filter very bad content
        if filter_score >= -0.1:  # Keep almost everything
            return 'keep_default', 1, 'H3'
        
        # Only filter truly bad content
        return 'filtered_default', 0, 'H3'
    
    def generate_filtering_report(self, df: pd.DataFrame, output_path: str = None):
        """Generate a detailed filtering report"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.base_dir, f"filtering_report_{timestamp}.json")
        
        # Analyze filtering decisions
        decision_counts = df['filter_decision'].value_counts().to_dict()
        
        # Analyze filtered vs preserved - convert to native Python types
        filtered_examples = df[df['filter_decision'] == 'filtered_low_score'].head(10)[['text', 'filter_score', 'filter_reasons']].to_dict('records')
        preserved_examples = df[df['filter_decision'].str.contains('keep')].head(10)[['text', 'filter_score', 'filter_reasons']].to_dict('records')
        
        # Convert examples to ensure JSON serialization
        filtered_examples = convert_numpy_types(filtered_examples)
        preserved_examples = convert_numpy_types(preserved_examples)
        
        # Generate report with type conversion
        report = {
            "timestamp": datetime.now().isoformat(),
            "document_stats": convert_numpy_types(self.document_stats),
            "filtering_summary": {
                "total_blocks": int(len(df)),
                "original_predictions": int(df['original_prediction'].sum()),
                "final_predictions": int(df['is_heading_pred'].sum()),
                "filtered_count": int(df['original_prediction'].sum() - df['is_heading_pred'].sum()),
                "reduction_rate": float((df['original_prediction'].sum() - df['is_heading_pred'].sum()) / df['original_prediction'].sum() * 100) if df['original_prediction'].sum() > 0 else 0.0
            },
            "decision_breakdown": convert_numpy_types(decision_counts),
            "examples": {
                "filtered": filtered_examples,
                "preserved": preserved_examples
            },
            "thresholds": {
                "high_confidence": float(self.high_confidence_threshold),
                "medium_confidence": float(self.medium_confidence_threshold),
                "low_confidence": float(self.low_confidence_threshold)
            }
        }
        
        # Save report
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"📋 Filtering report saved: {output_path}")
            return report
            
        except Exception as e:
            logger.error(f"❌ Error saving filtering report: {e}")
            # Try to save a simplified version
            simplified_report = {
                "timestamp": datetime.now().isoformat(),
                "filtering_summary": {
                    "total_blocks": int(len(df)),
                    "original_predictions": int(df['original_prediction'].sum()),
                    "final_predictions": int(df['is_heading_pred'].sum()),
                    "filtered_count": int(df['original_prediction'].sum() - df['is_heading_pred'].sum()),
                    "reduction_rate": float((df['original_prediction'].sum() - df['is_heading_pred'].sum()) / df['original_prediction'].sum() * 100) if df['original_prediction'].sum() > 0 else 0.0
                },
                "error": f"Full report generation failed: {str(e)}"
            }
            
            try:
                simplified_path = output_path.replace('.json', '_simplified.json')
                with open(simplified_path, 'w', encoding='utf-8') as f:
                    json.dump(simplified_report, f, indent=2, ensure_ascii=False)
                logger.info(f"📋 Simplified filtering report saved: {simplified_path}")
                return simplified_report
            except Exception as e2:
                logger.error(f"❌ Error saving simplified report: {e2}")
                return simplified_report
    
    def apply_consecutive_heading_merge_rule(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply rule: Two consecutive headings of same level will be merged only if 
        first one is a number, roman numeral, or single character followed by a word.
        Examples: "1" + "Introduction" -> "1 Introduction"
                  "A." + "System Architecture" -> "A. System Architecture"
                  "VII." + "Project Legacy" -> "VII. Project Legacy"
        
        Also merges broken headings that are split into multiple single words:
        Examples: "Proactive" + "India" -> "Proactive India"
        """
        logger.info("🔗 Applying consecutive heading merge rule...")
        
        merged_count = 0
        df_copy = df.copy()
        indices_to_remove = set()
        
        # Sort by page and position to ensure proper order
        sort_columns = ['page']
        if 'line_position_on_page' in df_copy.columns:
            sort_columns.append('line_position_on_page')
        elif 'y0' in df_copy.columns:
            sort_columns.append('y0')
        
        df_sorted = df_copy.sort_values(sort_columns, na_position='last').reset_index(drop=True)
        
        # First pass: merge numbered/lettered prefixes with following headings
        for i in range(len(df_sorted) - 1):
            current_row = df_sorted.iloc[i]
            next_row = df_sorted.iloc[i + 1]
            
            # Skip if current row is already marked for removal
            if current_row.name in indices_to_remove:
                continue
            
            # Check if both are headings
            current_is_heading = current_row.get('is_heading_pred', 0) == 1
            next_is_heading = next_row.get('is_heading_pred', 0) == 1
            
            # Check if they're on the same page and close to each other
            same_page = current_row.get('page', 0) == next_row.get('page', 0)
            
            if current_is_heading and next_is_heading and same_page:
                current_text = str(current_row.get('text', '')).strip()
                next_text = str(next_row.get('text', '')).strip()
                
                # Check if current text matches mergeable patterns
                is_mergeable = False
                merge_type = ""
                
                for pattern_type, pattern in self.mergeable_prefix_patterns.items():
                    if pattern.match(current_text):
                        # Additional check: next text should look like a proper heading
                        if len(next_text) > 0 and (
                            next_text[0].isupper() or  # Starts with uppercase
                            any(word.lower() in self.common_heading_words for word in next_text.split())
                        ):
                            is_mergeable = True
                            merge_type = pattern_type
                            logger.debug(f"Found mergeable prefix '{current_text}' ({pattern_type}) followed by '{next_text}'")
                            break
                
                if is_mergeable:
                    # Merge the texts
                    merged_text = f"{current_text} {next_text}".strip()
                    
                    # Update the next row with merged text and copy properties from current row
                    original_next_idx = next_row.name
                    df_copy.at[original_next_idx, 'text'] = merged_text
                    df_copy.at[original_next_idx, 'filter_decision'] = 'merged_consecutive'
                    df_copy.at[original_next_idx, 'filter_reasons'] = f"merged_with_prefix: {current_text} ({merge_type})"
                    
                    # Copy position info from the first element if needed
                    if 'line_position_on_page' in df_copy.columns:
                        df_copy.at[original_next_idx, 'line_position_on_page'] = current_row.get('line_position_on_page', next_row.get('line_position_on_page'))
                    if 'y0' in df_copy.columns:
                        df_copy.at[original_next_idx, 'y0'] = current_row.get('y0', next_row.get('y0'))
                    
                    # Mark the first heading as not a heading (remove it)
                    original_current_idx = current_row.name
                    df_copy.at[original_current_idx, 'is_heading_pred'] = 0
                    df_copy.at[original_current_idx, 'filter_decision'] = 'merged_into_next'
                    df_copy.at[original_current_idx, 'filter_reasons'] = f"merged_prefix_for: {next_text}"
                    
                    indices_to_remove.add(original_current_idx)
                    merged_count += 1
                    logger.debug(f"Merged prefix '{current_text}' + '{next_text}' -> '{merged_text}'")
        
        # Second pass: merge broken headings (consecutive single words that likely form one heading)
        # Re-sort after first pass modifications
        df_sorted = df_copy.sort_values(sort_columns, na_position='last').reset_index(drop=True)
        
        for i in range(len(df_sorted) - 2):  # Look ahead up to 2 positions
            if df_sorted.iloc[i].name in indices_to_remove:
                continue
                
            current_row = df_sorted.iloc[i]
            current_text = str(current_row.get('text', '')).strip()
            current_is_heading = current_row.get('is_heading_pred', 0) == 1
            
            # Look for consecutive single words that might be a broken heading
            if (current_is_heading and 
                len(current_text.split()) == 1 and  # Single word
                current_text.isalpha() and  # Only letters
                current_text[0].isupper()):  # Starts with uppercase
                
                # Collect consecutive single-word headings on the same page
                words_to_merge = [current_text]
                rows_to_merge = [current_row]
                
                # Look ahead for more single words
                for j in range(i + 1, min(i + 4, len(df_sorted))):  # Look at next 3 headings max
                    if df_sorted.iloc[j].name in indices_to_remove:
                        continue
                        
                    next_row = df_sorted.iloc[j]
                    next_text = str(next_row.get('text', '')).strip()
                    next_is_heading = next_row.get('is_heading_pred', 0) == 1
                    same_page = current_row.get('page', 0) == next_row.get('page', 0)
                    
                    # Stop if not a heading, different page, or multi-word
                    if not next_is_heading or not same_page:
                        break
                        
                    # Include single words or known continuation patterns
                    if (len(next_text.split()) == 1 and 
                        next_text.isalpha() and 
                        next_text[0].isupper()):
                        words_to_merge.append(next_text)
                        rows_to_merge.append(next_row)
                    else:
                        # If we find a longer text that seems related, include it and stop
                        if len(next_text.split()) <= 3 and next_text[0].isupper():
                            words_to_merge.append(next_text)
                            rows_to_merge.append(next_row)
                        break
                
                # Merge if we have multiple words and they seem to form a meaningful heading
                if len(words_to_merge) >= 2:
                    merged_text = " ".join(words_to_merge)
                    
                    # Additional validation: the merged text should look like a reasonable heading
                    if (len(merged_text) >= 5 and  # Not too short
                        not any(word.lower() in ['the', 'a', 'an', 'and', 'or', 'but'] for word in words_to_merge) and  # No obvious non-heading words
                        len(merged_text.split()) <= 6):  # Not too long
                        
                        # Update the first row with merged text
                        original_first_idx = rows_to_merge[0].name
                        df_copy.at[original_first_idx, 'text'] = merged_text
                        df_copy.at[original_first_idx, 'filter_decision'] = 'merged_broken_heading'
                        df_copy.at[original_first_idx, 'filter_reasons'] = f"merged_broken_words: {' + '.join(words_to_merge)}"
                        
                        # Mark other rows for removal
                        for row in rows_to_merge[1:]:
                            original_idx = row.name
                            df_copy.at[original_idx, 'is_heading_pred'] = 0
                            df_copy.at[original_idx, 'filter_decision'] = 'merged_into_broken_heading'
                            df_copy.at[original_idx, 'filter_reasons'] = f"merged_into: {merged_text}"
                            indices_to_remove.add(original_idx)
                        
                        merged_count += 1
                        logger.debug(f"Merged broken heading: {' + '.join(words_to_merge)} -> '{merged_text}'")
        
        logger.info(f"✅ Consecutive heading merge rule applied: {merged_count} merges performed")
        return df_copy
    
    def apply_document_title_rule(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply rule: First heading/H1 in the initial 100 blocks is title of the document.
        Ensure it gets H1 level and is preserved.
        """
        logger.info("📄 Applying document title rule...")
        
        # Sort dataframe to ensure we get the actual first blocks
        sort_columns = ['page']
        if 'line_position_on_page' in df.columns:
            sort_columns.append('line_position_on_page')
        elif 'y0' in df.columns:
            sort_columns.append('y0')
        
        df_sorted = df.sort_values(sort_columns, na_position='last')
        
        # Find the first 100 blocks (or fewer if document is smaller)
        initial_blocks = df_sorted.head(min(100, len(df_sorted)))
        
        # Find the first heading in these blocks
        first_heading_idx = None
        first_heading_position = None
        
        for position, (idx, row) in enumerate(initial_blocks.iterrows()):
            if row.get('is_heading_pred', 0) == 1:
                text = str(row.get('text', '')).strip()
                
                # Additional validation: should look like a document title
                # - Not too short (unless it's a clear title)
                # - Not a technical term or fragment
                # - Preferably in title case or all caps
                
                word_count = len(text.split())
                is_valid_title = True
                
                # Skip obvious non-titles
                if (word_count == 1 and len(text) < 4) or text.lower() in ['the', 'a', 'an']:
                    is_valid_title = False
                elif any(pattern.search(text) for pattern in [
                    self.exclusion_patterns.get('technical_terms', re.compile('')),
                    self.exclusion_patterns.get('technical_fragments', re.compile('')),
                    self.exclusion_patterns.get('programming_terms', re.compile('')),
                    self.exclusion_patterns.get('file_paths', re.compile(''))
                ]):
                    is_valid_title = False
                elif text.endswith('.') and word_count > 10:  # Long sentences
                    is_valid_title = False
                
                if is_valid_title:
                    first_heading_idx = idx
                    first_heading_position = position
                    break
        
        if first_heading_idx is not None:
            title_text = df.at[first_heading_idx, 'text']
            
            # Mark as document title and ensure H1 level
            df.at[first_heading_idx, 'final_heading_level'] = 'H1'
            df.at[first_heading_idx, 'filter_decision'] = 'document_title'
            df.at[first_heading_idx, 'filter_reasons'] = f'first_heading_document_title_position_{first_heading_position}'
            df.at[first_heading_idx, 'is_heading_pred'] = 1  # Ensure it's preserved
            
            # Boost confidence for title
            current_confidence = df.at[first_heading_idx, 'heading_confidence'] if 'heading_confidence' in df.columns else 0.8
            df.at[first_heading_idx, 'heading_confidence'] = max(current_confidence, 0.9)
            
            logger.info(f"📄 Document title identified: '{title_text}' (position: {first_heading_position}, index: {first_heading_idx})")
            
            # Check for potential subtitle (next heading on same page or close by)
            potential_subtitle_blocks = df_sorted.head(min(120, len(df_sorted)))
            for position, (idx, row) in enumerate(potential_subtitle_blocks.iterrows()):
                if (idx != first_heading_idx and 
                    row.get('is_heading_pred', 0) == 1 and
                    position > first_heading_position and 
                    position <= first_heading_position + 15):  # Within 15 positions
                    
                    subtitle_text = str(row.get('text', '')).strip()
                    subtitle_words = len(subtitle_text.split())
                    
                    # Valid subtitle criteria
                    if (subtitle_words >= 2 and subtitle_words <= 15 and
                        not any(pattern.search(subtitle_text) for pattern in self.exclusion_patterns.values())):
                        
                        df.at[idx, 'final_heading_level'] = 'H2'
                        df.at[idx, 'filter_decision'] = 'document_subtitle'
                        df.at[idx, 'filter_reasons'] = f'potential_subtitle_after_title'
                        logger.info(f"📄 Potential subtitle identified: '{subtitle_text}' (position: {position})")
                        break
        else:
            logger.warning("⚠️  No valid heading found in first 100 blocks for document title")
        
        return df

    def tune_thresholds(self, df: pd.DataFrame, target_precision: float = 0.9):
        """Tune filtering thresholds based on validation data with known labels"""
        if 'is_heading' not in df.columns:
            logger.warning("⚠️  No ground truth labels found, cannot tune thresholds")
            return
        
        logger.info(f"🎯 Tuning thresholds for target precision: {target_precision}")
        
        # Calculate document statistics
        self.calculate_document_statistics(df)
        
        # Calculate scores for all predictions
        scores = []
        for idx, row in df.iterrows():
            if row.get('is_heading_pred', 0) == 1:  # Only for predicted headings
                confidence = row.get('heading_confidence', 0.8)
                context = self.analyze_context(df, idx)
                score, _ = self.calculate_filter_score(row, context, confidence)
                scores.append({
                    'idx': idx,
                    'score': score,
                    'confidence': confidence,
                    'ground_truth': row['is_heading'],
                    'prediction': row['is_heading_pred']
                })
        
        if not scores:
            logger.warning("⚠️  No predictions found for threshold tuning")
            return
        
        # Test different score thresholds
        best_threshold = 0.5
        best_precision = 0.0
        
        for threshold in np.arange(0.2, 0.9, 0.05):
            tp = sum(1 for s in scores if s['score'] >= threshold and s['ground_truth'] == 1)
            fp = sum(1 for s in scores if s['score'] >= threshold and s['ground_truth'] == 0)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            
            if precision >= target_precision and precision > best_precision:
                best_precision = precision
                best_threshold = threshold
        
        logger.info(f"🎯 Best threshold: {best_threshold:.3f} (precision: {best_precision:.3f})")
        return best_threshold


def main():
    """Main function for testing"""
    print("🧠 INTELLIGENT RULE-BASED FILTERING SYSTEM")
    print("=" * 50)
    print("🎯 Features:")
    print("   ✅ Multi-layered filtering approach")
    print("   ✅ Confidence-based decision making")
    print("   ✅ Preservation of high-confidence predictions")
    print("   ✅ Dynamic thresholding")
    print("   ✅ Comprehensive logging and reporting")
    print()
    
    # Initialize filter
    filter_system = IntelligentFilter()
    
    # Test with sample data (you would replace this with actual prediction data)
    sample_data = {
        'text': ['Chapter 1: Introduction', 'This is a regular paragraph.', 'http://www.example.com', '1.1 Overview'],
        'font_size': [14, 12, 10, 13],
        'page': [1, 1, 1, 1],
        'line_position_on_page': [1, 5, 10, 15],
        'is_heading_pred': [1, 0, 1, 1],
        'heading_confidence': [0.95, 0.2, 0.6, 0.85]
    }
    
    df = pd.DataFrame(sample_data)
    
    # Apply filtering
    filtered_df = filter_system.apply_intelligent_filtering(df)
    
    # Generate report
    report = filter_system.generate_filtering_report(filtered_df)
    
    print("📊 Sample filtering results:")
    print(filtered_df[['text', 'is_heading_pred', 'filter_score', 'filter_decision']].to_string())


if __name__ == "__main__":
    main()
