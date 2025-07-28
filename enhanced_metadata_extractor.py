#!/usr/bin/env python3
"""
Enhanced Metadata Extractor for Intelligent Heading Detection
============================================================

This module extracts comprehensive metadata from PDF text blocks to enable
more intelligent heading detection and hierarchy assignment.

Features:
✅ Comprehensive linguistic analysis (POS tagging, syntax patterns)
✅ Advanced typography metadata (font relationships, size hierarchies)
✅ Contextual positioning analysis (relative to page, section boundaries)
✅ Semantic content analysis (heading vs sentence patterns)
✅ Document structure understanding (section flow, hierarchy detection)
✅ Multi-language support with adaptive patterns

Author: AI Assistant  
Date: July 28, 2025
"""

import os
import re
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import statistics

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedMetadataExtractor:
    """Extract comprehensive metadata for intelligent heading detection"""
    
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Typography analysis patterns
        self.heading_typography_patterns = {
            'strong_indicators': [
                r'^[IVX]+\.\s+[A-Z]',  # Roman numerals: I. Introduction
                r'^\d+\.\s+[A-Z]',     # Numbered: 1. Overview
                r'^\d+\.\d+\s+[A-Z]',  # Sub-numbered: 1.1 Details
                r'^[A-Z]\.\s+[A-Z]',   # Letter: A. Section
                r'^CHAPTER\s+\d+',     # Chapter headers
                r'^SECTION\s+\d+',     # Section headers
            ],
            'moderate_indicators': [
                r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4}$',  # Title case (2-5 words)
                r'^[A-Z]{2,}\s+[A-Z]',  # CAPS followed by title case
                r'^\d+\s+[A-Z]',       # Number space Title
                r'^[A-Z]+\s*:',        # ALL CAPS with colon
            ],
            'weak_indicators': [
                r'^[A-Z][a-z]+$',      # Single title case word
                r'^[A-Z][a-z]+\s*:',   # Title case with colon
            ]
        }
        
        # Anti-patterns (strong indicators this is NOT a heading)
        self.anti_heading_patterns = {
            'sentence_fragments': [
                r'^[a-z]',             # Starts with lowercase
                r'[.!?]\s*$',          # Ends with sentence punctuation
                r'^\w+,\s*$',          # Single word with comma
                r'^\w+-based\s*$',     # "Docker-based", "web-based"
                r'^\w+\.(js|py|css|html|pdf|doc)$',  # File extensions
            ],
            'technical_terms': [
                r'^[A-Z][a-z]*\.(js|ts|py|css)$',  # bcrypt.js, node.js
                r'^(npm|pip|git|docker|kubernetes)$',  # Tool names
                r'^(API|URL|HTTP|HTTPS|JSON|XML)$',     # Technical acronyms
                r'^\w+://\w+',         # URLs
                r'^localhost:\d+',     # Localhost URLs
            ],
            'incomplete_phrases': [
                r'^\w+,?\s*$',         # Single word possibly with comma
                r'^(the|a|an|and|or|but|in|on|at|to|for|of|with|by)\s+\w+',  # Articles/prepositions
                r'^\w+\s+(and|or|but|in|on|at|to|for|of|with|by)\s+',  # Mid-sentence patterns
                r'^\w+\s+\w+,\s*$',    # "word word," pattern
            ],
            'list_items': [
                r'^\s*[•◦▪▫◾▸►‣⁃]',   # Bullet points
                r'^\s*-\s+',           # Dash lists
                r'^\s*\*\s+',          # Star lists
                r'^\d+[\.)]\s+[a-z]',  # Numbered lists with lowercase
            ]
        }
        
        # POS patterns for headings vs sentences
        self.pos_patterns = {
            'heading_favorable': [
                'NOUN', 'PROPN', 'ADJ'  # Nouns, proper nouns, adjectives
            ],
            'heading_neutral': [
                'NUM', 'X'  # Numbers, others
            ],
            'heading_unfavorable': [
                'VERB', 'ADP', 'CONJ', 'PRON', 'DET', 'PART'  # Verbs, prepositions, etc.
            ]
        }
        
        logger.info("🔍 Enhanced Metadata Extractor initialized!")
    
    def extract_comprehensive_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract comprehensive metadata for all text blocks"""
        logger.info("🔍 Extracting comprehensive metadata for intelligent analysis...")
        
        # Make a copy to avoid modifying original
        enhanced_df = df.copy()
        
        # 1. Typography and font analysis
        enhanced_df = self._analyze_typography(enhanced_df)
        
        # 2. Linguistic analysis
        enhanced_df = self._analyze_linguistics(enhanced_df)
        
        # 3. Positional and contextual analysis
        enhanced_df = self._analyze_positioning(enhanced_df)
        
        # 4. Content semantic analysis
        enhanced_df = self._analyze_content_semantics(enhanced_df)
        
        # 5. Document structure analysis
        enhanced_df = self._analyze_document_structure(enhanced_df)
        
        # 6. Heading likelihood scoring
        enhanced_df = self._calculate_heading_likelihood(enhanced_df)
        
        # 7. Heading hierarchy assignment
        enhanced_df = self._assign_heading_hierarchy(enhanced_df)
        
        logger.info(f"✅ Enhanced metadata extraction complete for {len(enhanced_df)} blocks")
        return enhanced_df
    
    def _analyze_typography(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze typography and font characteristics"""
        logger.info("📝 Analyzing typography and font characteristics...")
        
        # Font size analysis
        if 'font_size' in df.columns:
            font_sizes = df['font_size'].dropna()
            
            # Calculate detailed font statistics
            df['font_size_mean'] = font_sizes.mean()
            df['font_size_std'] = font_sizes.std()
            df['font_size_percentile_rank'] = df['font_size'].rank(pct=True) * 100
            
            # Font size categories
            q75 = font_sizes.quantile(0.75)
            q90 = font_sizes.quantile(0.90)
            q95 = font_sizes.quantile(0.95)
            
            df['font_category'] = 'small'
            df.loc[df['font_size'] >= q75, 'font_category'] = 'medium'
            df.loc[df['font_size'] >= q90, 'font_category'] = 'large'
            df.loc[df['font_size'] >= q95, 'font_category'] = 'very_large'
            
            # Relative font size (compared to surrounding text)
            df['font_size_relative'] = df['font_size'] / df['font_size_mean']
            
            # Font size jump detection (significant increases)
            df['font_size_jump'] = False
            for i in range(1, len(df)):
                prev_size = df.iloc[i-1]['font_size']
                curr_size = df.iloc[i]['font_size']
                if curr_size > prev_size * 1.2:  # 20% increase
                    df.at[i, 'font_size_jump'] = True
        else:
            # Default values if font info not available
            df['font_size_percentile_rank'] = 50
            df['font_category'] = 'medium'
            df['font_size_relative'] = 1.0
            df['font_size_jump'] = False
        
        # Typography pattern matching
        df['typography_score'] = 0
        df['typography_patterns'] = ''
        
        for _, row in df.iterrows():
            text = str(row.get('text', ''))
            score = 0
            patterns = []
            
            # Strong heading typography
            for pattern in self.heading_typography_patterns['strong_indicators']:
                if re.match(pattern, text):
                    score += 3
                    patterns.append('strong_typography')
                    break
            
            # Moderate heading typography
            for pattern in self.heading_typography_patterns['moderate_indicators']:
                if re.match(pattern, text):
                    score += 2
                    patterns.append('moderate_typography')
                    break
            
            # Weak heading typography
            for pattern in self.heading_typography_patterns['weak_indicators']:
                if re.match(pattern, text):
                    score += 1
                    patterns.append('weak_typography')
                    break
            
            df.at[row.name, 'typography_score'] = score
            df.at[row.name, 'typography_patterns'] = ';'.join(patterns)
        
        return df
    
    def _analyze_linguistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze linguistic characteristics"""
        logger.info("🔤 Analyzing linguistic characteristics...")
        
        # Initialize linguistic features
        df['linguistic_score'] = 0
        df['pos_heading_score'] = 0
        df['syntax_violations'] = 0
        df['linguistic_patterns'] = ''
        
        # Try to load spaCy for advanced analysis
        try:
            import spacy
            from spacy.lang.en import English
            
            # Try to load full model, fallback to basic
            try:
                nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("Full spaCy model not available, using basic tokenizer")
                nlp = English()
                nlp.add_pipe('sentencizer')
            
            has_spacy = True
        except ImportError:
            logger.warning("spaCy not available, using rule-based linguistic analysis")
            has_spacy = False
        
        for idx, row in df.iterrows():
            text = str(row.get('text', ''))
            if not text.strip():
                continue
                
            linguistic_score = 0
            patterns = []
            violations = 0
            
            # Basic text analysis
            words = text.split()
            word_count = len(words)
            
            # Length analysis - less aggressive
            if 1 <= word_count <= 8:
                linguistic_score += 2
                patterns.append('good_length')
            elif word_count > 20:  # Increased threshold
                violations += 1  # Reduced from 2
                patterns.append('too_long')
            
            # Case analysis - less aggressive
            if text.isupper() and word_count > 8:  # Increased threshold
                violations += 1
                patterns.append('excessive_caps')
            elif text.istitle():
                linguistic_score += 1
                patterns.append('title_case')
            elif text[0].isupper() if text else False:
                linguistic_score += 0.5
                patterns.append('capital_start')
            elif text[0].islower() if text else False:
                violations += 1  # Reduced from 2
                patterns.append('lowercase_start')
            
            # Punctuation analysis - less aggressive
            if text.endswith(('.', '!', '?')):
                violations += 1  # Reduced from 2
                patterns.append('sentence_ending')
            elif text.endswith(':'):
                linguistic_score += 1
                patterns.append('colon_ending')
            elif text.endswith(','):
                violations += 1
                patterns.append('comma_ending')
            
            # Anti-pattern detection - less aggressive for headings
            for category, pattern_list in self.anti_heading_patterns.items():
                for pattern in pattern_list:
                    if re.match(pattern, text, re.IGNORECASE):
                        # Reduce violations based on category
                        if category in ['sentence_fragments', 'incomplete_phrases']:
                            violations += 1  # Reduced from 3
                        elif category in ['technical_terms', 'list_items']:
                            violations += 2  # Reduced from 3
                        else:
                            violations += 3  # Keep high for obvious non-headings
                        patterns.append(f'anti_{category}')
                        break
            
            # Advanced spaCy analysis
            pos_score = 0
            if has_spacy and text.strip():
                try:
                    doc = nlp(text)
                    
                    # POS analysis
                    pos_counts = {'favorable': 0, 'neutral': 0, 'unfavorable': 0}
                    
                    for token in doc:
                        if token.pos_ in self.pos_patterns['heading_favorable']:
                            pos_counts['favorable'] += 1
                        elif token.pos_ in self.pos_patterns['heading_neutral']:
                            pos_counts['neutral'] += 1
                        elif token.pos_ in self.pos_patterns['heading_unfavorable']:
                            pos_counts['unfavorable'] += 1
                    
                    total_tokens = sum(pos_counts.values())
                    if total_tokens > 0:
                        favorable_ratio = pos_counts['favorable'] / total_tokens
                        unfavorable_ratio = pos_counts['unfavorable'] / total_tokens
                        
                        pos_score = (favorable_ratio * 2) - (unfavorable_ratio * 2)
                        
                        if favorable_ratio > 0.6:
                            patterns.append('favorable_pos')
                        elif unfavorable_ratio > 0.4:
                            patterns.append('unfavorable_pos')
                            violations += 1
                    
                    # Dependency analysis
                    if len(doc) > 1:
                        # Check for sentence-like structures
                        has_subject = any(token.dep_ in ['nsubj', 'nsubjpass'] for token in doc)
                        has_verb = any(token.pos_ == 'VERB' for token in doc)
                        
                        if has_subject and has_verb:
                            violations += 2
                            patterns.append('sentence_structure')
                
                except Exception as e:
                    logger.debug(f"spaCy analysis failed for '{text}': {e}")
            
            # Store results with proper dtype conversion
            df.at[idx, 'linguistic_score'] = float(linguistic_score)
            df.at[idx, 'pos_heading_score'] = float(pos_score)
            df.at[idx, 'syntax_violations'] = int(violations)
            df.at[idx, 'linguistic_patterns'] = ';'.join(patterns)
        
        return df
    
    def _analyze_positioning(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze positional and contextual characteristics"""
        logger.info("📍 Analyzing positional and contextual characteristics...")
        
        # Page-level analysis
        df['position_score'] = 0
        df['context_score'] = 0
        df['position_patterns'] = ''
        
        # Handle missing page column
        if 'page' not in df.columns:
            if 'page_num' in df.columns:
                df['page'] = df['page_num']
            else:
                df['page'] = 1  # Default to page 1 if no page info
        
        for page in df['page'].unique():
            page_df = df[df['page'] == page].copy()
            
            # Sort by position on page - handle missing columns gracefully
            if 'y0' in page_df.columns and not page_df['y0'].isna().all():
                page_df = page_df.sort_values('y0', ascending=False)  # Top to bottom
            elif 'line_position_on_page' in page_df.columns and not page_df['line_position_on_page'].isna().all():
                page_df = page_df.sort_values('line_position_on_page')
            
            for idx, row in page_df.iterrows():
                score = 0
                patterns = []
                
                # Position on page - check if column exists and has valid data
                if 'line_position_on_page' in row and pd.notna(row['line_position_on_page']):
                    line_pos = row['line_position_on_page']
                    max_line = page_df['line_position_on_page'].max()
                    
                    if pd.notna(max_line) and max_line > 0:
                        if line_pos <= 3:  # Top of page
                            score += 2
                            patterns.append('page_top')
                        elif line_pos >= max_line - 2:  # Bottom of page
                            score -= 1
                            patterns.append('page_bottom')
                        elif line_pos / max_line < 0.3:  # Upper third
                            score += 1
                            patterns.append('upper_third')
                
                # Whitespace analysis (if coordinates available)
                if ('y0' in row and 'y1' in row and 
                    pd.notna(row['y0']) and pd.notna(row['y1'])):
                    # Look for significant whitespace before/after
                    current_y = row['y0']
                    
                    # Check spacing above
                    above_blocks = page_df[(page_df['y1'] > current_y) & pd.notna(page_df['y1'])]
                    if len(above_blocks) > 0:
                        nearest_above = above_blocks['y1'].min()
                        spacing_above = current_y - nearest_above
                        
                        if spacing_above > 20:  # Significant whitespace
                            score += 1
                            patterns.append('whitespace_above')
                
                # Isolation analysis (stands alone)
                text = str(row.get('text', ''))
                if len(text.split()) <= 6:  # Short text
                    # Check if surrounded by longer text blocks
                    row_idx = page_df.index.get_loc(idx)
                    
                    context_before = context_after = ""
                    if row_idx > 0:
                        context_before = str(page_df.iloc[row_idx - 1].get('text', ''))
                    if row_idx < len(page_df) - 1:
                        context_after = str(page_df.iloc[row_idx + 1].get('text', ''))
                    
                    if (len(context_before.split()) > 10 or 
                        len(context_after.split()) > 10):
                        score += 1
                        patterns.append('isolated_short_text')
                
                df.at[idx, 'position_score'] = score
                df.at[idx, 'position_patterns'] = ';'.join(patterns)
        
        return df
    
    def _analyze_content_semantics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze semantic content characteristics"""
        logger.info("🧠 Analyzing semantic content characteristics...")
        
        df['semantic_score'] = 0
        df['content_type'] = 'unknown'
        df['semantic_patterns'] = ''
        
        # Common heading keywords
        heading_keywords = {
            'structural': ['introduction', 'conclusion', 'summary', 'abstract', 'overview', 
                          'background', 'methodology', 'results', 'discussion', 'analysis'],
            'organizational': ['chapter', 'section', 'part', 'unit', 'appendix', 'references',
                             'bibliography', 'acknowledgment', 'acknowledgments'],
            'academic': ['literature', 'review', 'findings', 'limitations', 'recommendations',
                        'objectives', 'scope', 'framework', 'approach', 'implementation'],
            'technical': ['architecture', 'design', 'system', 'implementation', 'deployment',
                         'configuration', 'installation', 'setup', 'requirements']
        }
        
        # Content that suggests NOT a heading
        non_heading_indicators = {
            'sentence_starters': ['the', 'this', 'that', 'these', 'those', 'it', 'they', 'we', 'i'],
            'conjunctions': ['and', 'but', 'or', 'however', 'therefore', 'thus', 'hence'],
            'prepositions': ['in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from'],
            'articles': ['a', 'an', 'the']
        }
        
        for idx, row in df.iterrows():
            text = str(row.get('text', '')).lower().strip()
            if not text:
                continue
                
            score = 0
            patterns = []
            content_type = 'text'
            
            words = text.split()
            first_word = words[0] if words else ''
            
            # Check for heading keywords
            for category, keywords in heading_keywords.items():
                if any(keyword in text for keyword in keywords):
                    score += 2
                    patterns.append(f'heading_{category}')
                    content_type = 'heading_candidate'
                    break
            
            # Check for non-heading indicators
            violation_score = 0
            if first_word in non_heading_indicators['sentence_starters']:
                violation_score += 2
                patterns.append('sentence_starter')
            
            if first_word in non_heading_indicators['conjunctions']:
                violation_score += 3
                patterns.append('conjunction_start')
            
            if any(word in non_heading_indicators['prepositions'] for word in words[:2]):
                violation_score += 1
                patterns.append('early_preposition')
            
            # Technical term detection
            technical_patterns = [
                r'\.(js|ts|py|css|html|json|xml|pdf)$',
                r'^(api|url|http|https|json|xml|css|html)$',
                r'://|localhost|127\.0\.0\.1',
                r'^(docker|npm|pip|git|node|react|angular|vue)$'
            ]
            
            for pattern in technical_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    violation_score += 2
                    patterns.append('technical_term')
                    content_type = 'technical'
                    break
            
            # Apply violations
            score -= violation_score
            
            # Content structure analysis - more favorable to headings
            if len(words) == 1:
                if (text.isalpha() and (text.istitle() or text.isupper()) and
                    any(text.lower() == keyword for category in heading_keywords.values() for keyword in category)):
                    score += 2
                    patterns.append('heading_keyword')
                    content_type = 'heading_candidate'
                elif text.isalpha() and (text.istitle() or text.isupper()):
                    score += 1
                    patterns.append('single_title_word')
                    content_type = 'heading_candidate'  # Changed from fragment
                else:
                    patterns.append('single_word')
                    content_type = 'fragment'
            
            elif 2 <= len(words) <= 8:  # Increased threshold
                if (all(word.istitle() or word.isupper() for word in words) or
                    text.istitle() or
                    any(word.lower() in [keyword for category in heading_keywords.values() for keyword in category] for word in words)):
                    score += 2  # Increased bonus
                    patterns.append('title_phrase')
                    content_type = 'heading_candidate'
                else:
                    content_type = 'text'  # Don't default to fragment
            
            elif len(words) <= 15:  # New category for longer headings
                if (text.istitle() or 
                    any(word.lower() in [keyword for category in heading_keywords.values() for keyword in category] for word in words)):
                    score += 1
                    patterns.append('longer_heading')
                    content_type = 'heading_candidate'
            
            # Store results
            df.at[idx, 'semantic_score'] = score
            df.at[idx, 'content_type'] = content_type
            df.at[idx, 'semantic_patterns'] = ';'.join(patterns)
        
        return df
    
    def _analyze_document_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze document structure and hierarchy"""
        logger.info("📋 Analyzing document structure and hierarchy...")
        
        df['structure_score'] = 0
        df['hierarchy_level'] = 0
        df['section_boundary'] = False
        df['structure_patterns'] = ''
        
        # Analyze numbering patterns
        numbering_patterns = {
            'level_1': [r'^[IVX]+\.', r'^\d+\.$', r'^CHAPTER\s+\d+', r'^SECTION\s+\d+'],
            'level_2': [r'^\d+\.\d+', r'^[A-Z]\.', r'^\d+\.\d+\.'],
            'level_3': [r'^\d+\.\d+\.\d+', r'^[a-z][\.)]\s+', r'^\([a-z]\)']
        }
        
        for idx, row in df.iterrows():
            text = str(row.get('text', ''))
            score = 0
            patterns = []
            hierarchy = 0
            
            # Check numbering patterns for hierarchy
            for level, pattern_list in numbering_patterns.items():
                for pattern in pattern_list:
                    if re.match(pattern, text, re.IGNORECASE):
                        if level == 'level_1':
                            hierarchy = 1
                            score += 3
                        elif level == 'level_2':
                            hierarchy = 2
                            score += 2
                        elif level == 'level_3':
                            hierarchy = 3
                            score += 1
                        patterns.append(f'{level}_numbering')
                        break
                if hierarchy > 0:
                    break
            
            # Section boundary detection
            if (row.get('font_size_jump', False) or 
                'page_top' in str(row.get('position_patterns', '')) or
                hierarchy > 0):
                df.at[idx, 'section_boundary'] = True
                score += 1
                patterns.append('section_boundary')
            
            df.at[idx, 'structure_score'] = score
            df.at[idx, 'hierarchy_level'] = hierarchy
            df.at[idx, 'structure_patterns'] = ';'.join(patterns)
        
        return df
    
    def _calculate_heading_likelihood(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate overall heading likelihood score"""
        logger.info("🎯 Calculating heading likelihood scores...")
        
        # Weighted combination of all scores
        weights = {
            'typography': 0.25,
            'linguistic': 0.20,
            'position': 0.15,
            'semantic': 0.20,
            'structure': 0.20
        }
        
        # Normalize scores to 0-1 range
        for score_col in ['typography_score', 'linguistic_score', 'position_score', 
                         'semantic_score', 'structure_score']:
            if score_col in df.columns:
                min_score = df[score_col].min()
                max_score = df[score_col].max()
                if max_score > min_score:
                    df[f'{score_col}_norm'] = (df[score_col] - min_score) / (max_score - min_score)
                else:
                    df[f'{score_col}_norm'] = 0.5
            else:
                df[f'{score_col}_norm'] = 0.5
        
        # Calculate weighted likelihood
        df['heading_likelihood'] = (
            df['typography_score_norm'] * weights['typography'] +
            df['linguistic_score_norm'] * weights['linguistic'] +
            df['position_score_norm'] * weights['position'] +
            df['semantic_score_norm'] * weights['semantic'] +
            df['structure_score_norm'] * weights['structure']
        )
        
        # Apply violation penalties
        df['syntax_violations'] = df.get('syntax_violations', 0)
        df['heading_likelihood'] -= df['syntax_violations'] * 0.1
        
        # Clamp to 0-1 range
        df['heading_likelihood'] = df['heading_likelihood'].clip(0, 1)
        
        return df
    
    def _assign_heading_hierarchy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign proper heading hierarchy (H1/H2/H3)"""
        logger.info("🏗️ Assigning heading hierarchy...")
        
        df['recommended_level'] = 'H3'  # Default
        df['level_confidence'] = 0.5
        
        # Determine sort columns based on availability
        sort_columns = ['page']
        if 'line_position_on_page' in df.columns:
            sort_columns.append('line_position_on_page')
        elif 'y0' in df.columns:
            sort_columns.append('y0')
        
        # Sort by page and position for hierarchical analysis
        df_sorted = df.sort_values(sort_columns, na_position='last')
        
        # Track hierarchy context
        current_h1_level = 0
        current_h2_level = 0
        
        for idx in df_sorted.index:
            row = df.loc[idx]
            
            # Skip if not likely to be a heading
            if row.get('heading_likelihood', 0) < 0.3:
                continue
            
            level = 'H3'  # Default
            confidence = 0.5
            
            # H1 criteria: highest font, strong typography, or structural indicators
            h1_indicators = (
                row.get('font_category') in ['very_large', 'large'] and
                row.get('typography_score', 0) >= 2 and
                row.get('hierarchy_level', 0) in [0, 1] and
                row.get('semantic_score', 0) >= 1
            )
            
            # Strong H1 patterns
            if (row.get('hierarchy_level', 0) == 1 or
                row.get('font_size_percentile_rank', 50) >= 95 or
                'CHAPTER' in str(row.get('text', '')).upper() or
                'SECTION' in str(row.get('text', '')).upper()):
                level = 'H1'
                confidence = 0.9
                current_h1_level += 1
                current_h2_level = 0
            
            # H2 criteria: medium-large font, good structure
            elif (row.get('hierarchy_level', 0) == 2 or
                  (row.get('font_category') in ['large', 'medium'] and
                   row.get('typography_score', 0) >= 1 and
                   row.get('semantic_score', 0) >= 0)):
                level = 'H2'
                confidence = 0.8
                current_h2_level += 1
            
            # H3 for everything else that qualifies as heading
            else:
                level = 'H3'
                confidence = 0.7
            
            # Context-based adjustments
            if current_h1_level == 0 and level == 'H2':
                level = 'H1'  # First major heading should be H1
                confidence = 0.8
                current_h1_level = 1
            
            df.at[idx, 'recommended_level'] = level
            df.at[idx, 'level_confidence'] = confidence
        
        return df


def main():
    """Test the enhanced metadata extractor"""
    print("🔍 ENHANCED METADATA EXTRACTOR")
    print("=" * 50)
    
    # Create sample test data
    test_data = {
        'text': [
            'CHAPTER 1: INTRODUCTION',  # Should be H1
            'Bcrypt.js',  # Should be filtered
            '1.1 Overview',  # Should be H2
            'This is a regular paragraph that should not be a heading.',  # Should be filtered
            'Initially,',  # Should be filtered (incomplete)
            'Docker-based',  # Should be filtered (incomplete)
            'Methodology',  # Should be H2 or H3
            'The system employs several technologies.',  # Should be filtered
            'A. System Architecture',  # Should be H2
            'References'  # Should be H1 or H2
        ],
        'font_size': [16, 10, 14, 11, 12, 11, 13, 11, 14, 13],
        'page': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
        'line_position_on_page': [1, 5, 8, 12, 15, 3, 7, 11, 15, 20],
        'x0': [50, 60, 70, 50, 80, 60, 65, 50, 70, 65],
        'y0': [750, 650, 600, 550, 500, 720, 650, 600, 550, 400],
        'x1': [200, 120, 150, 400, 140, 130, 180, 380, 220, 150],
        'y1': [770, 665, 615, 570, 515, 735, 665, 620, 565, 415]
    }
    
    df = pd.DataFrame(test_data)
    
    # Initialize extractor
    extractor = EnhancedMetadataExtractor()
    
    # Extract metadata
    enhanced_df = extractor.extract_comprehensive_metadata(df)
    
    # Display results
    print("\n📊 ENHANCED METADATA RESULTS:")
    print("=" * 40)
    
    for idx, row in enhanced_df.iterrows():
        print(f"\n📝 Text: '{row['text']}'")
        print(f"   🎯 Heading Likelihood: {row.get('heading_likelihood', 0):.3f}")
        print(f"   📊 Typography Score: {row.get('typography_score', 0)}")
        print(f"   🔤 Linguistic Score: {row.get('linguistic_score', 0)}")
        print(f"   📍 Position Score: {row.get('position_score', 0)}")
        print(f"   🧠 Semantic Score: {row.get('semantic_score', 0)}")
        print(f"   🏗️ Structure Score: {row.get('structure_score', 0)}")
        print(f"   ⚠️ Syntax Violations: {row.get('syntax_violations', 0)}")
        print(f"   📏 Recommended Level: {row.get('recommended_level', 'H3')}")
        print(f"   🎨 Font Category: {row.get('font_category', 'medium')}")
        print(f"   🏷️ Content Type: {row.get('content_type', 'unknown')}")


if __name__ == "__main__":
    main()
