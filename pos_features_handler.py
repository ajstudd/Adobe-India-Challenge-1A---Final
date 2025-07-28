#!/usr/bin/env python3
"""
POS Features Handler
===================

Robust handler for Part-of-Speech features with fallback mechanisms.
Ensures the system works with or without spaCy models.

Author: AI Assistant
Date: July 28, 2025
"""

import os
import sys
import logging
import subprocess
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class POSFeaturesHandler:
    """Handle POS features with robust fallback mechanisms"""
    
    def __init__(self):
        self.spacy_available = False
        self.nlp = None
        self.initialize_spacy()
    
    def initialize_spacy(self):
        """Initialize spaCy with automatic installation if needed"""
        try:
            import spacy
            
            # Try to load the model
            try:
                self.nlp = spacy.load("en_core_web_sm")
                self.spacy_available = True
                logger.info("✅ spaCy English model loaded successfully")
                return True
            except OSError:
                logger.warning("⚠️ spaCy English model not found. Attempting to install...")
                
                # Try to install the model
                try:
                    subprocess.check_call([
                        sys.executable, "-m", "spacy", "download", "en_core_web_sm"
                    ], capture_output=True)
                    
                    # Try loading again
                    self.nlp = spacy.load("en_core_web_sm")
                    self.spacy_available = True
                    logger.info("✅ spaCy English model installed and loaded successfully")
                    return True
                    
                except subprocess.CalledProcessError as e:
                    logger.warning(f"⚠️ Failed to install spaCy model automatically: {e}")
                    self.spacy_available = False
                    return False
                except Exception as e:
                    logger.warning(f"⚠️ Error during spaCy model installation: {e}")
                    self.spacy_available = False
                    return False
                    
        except ImportError:
            logger.warning("⚠️ spaCy not installed. POS features will use fallback method.")
            self.spacy_available = False
            return False
    
    def calculate_pos_features_spacy(self, df):
        """Calculate POS features using spaCy"""
        if not self.spacy_available or self.nlp is None:
            return self.calculate_pos_features_fallback(df)
        
        logger.info("🏷️ Calculating POS features using spaCy...")
        
        # Initialize POS feature columns
        pos_features = ['num_nouns', 'num_verbs', 'num_adjs', 'num_advs', 'num_propn', 'num_pronouns', 'num_other_pos']
        for feature in pos_features:
            if feature not in df.columns:
                df[feature] = 0
        
        # Process each text row
        for idx, row in df.iterrows():
            text = str(row.get('text', ''))
            if not text.strip():
                continue
            
            try:
                # Process with spaCy
                doc = self.nlp(text[:1000])  # Limit text length for performance
                
                # Count POS tags
                pos_counts = {
                    'num_nouns': 0,
                    'num_verbs': 0,
                    'num_adjs': 0,
                    'num_advs': 0,
                    'num_propn': 0,
                    'num_pronouns': 0,
                    'num_other_pos': 0
                }
                
                for token in doc:
                    if token.pos_ in ['NOUN']:
                        pos_counts['num_nouns'] += 1
                    elif token.pos_ in ['VERB']:
                        pos_counts['num_verbs'] += 1
                    elif token.pos_ in ['ADJ']:
                        pos_counts['num_adjs'] += 1
                    elif token.pos_ in ['ADV']:
                        pos_counts['num_advs'] += 1
                    elif token.pos_ in ['PROPN']:
                        pos_counts['num_propn'] += 1
                    elif token.pos_ in ['PRON']:
                        pos_counts['num_pronouns'] += 1
                    else:
                        pos_counts['num_other_pos'] += 1
                
                # Update dataframe
                for feature, count in pos_counts.items():
                    df.at[idx, feature] = count
                    
            except Exception as e:
                logger.warning(f"⚠️ Error processing text at index {idx}: {e}")
                continue
        
        logger.info("✅ POS features calculated successfully using spaCy")
        return df
    
    def calculate_pos_features_fallback(self, df):
        """Calculate POS features using simple pattern matching (fallback)"""
        logger.info("🏷️ Calculating POS features using fallback method...")
        
        # Initialize POS feature columns
        pos_features = ['num_nouns', 'num_verbs', 'num_adjs', 'num_advs', 'num_propn', 'num_pronouns', 'num_other_pos']
        for feature in pos_features:
            if feature not in df.columns:
                df[feature] = 0
        
        # Simple pattern-based POS estimation
        import re
        
        # Common patterns (very basic)
        noun_patterns = [
            r'\b\w+tion\b', r'\b\w+ness\b', r'\b\w+ment\b', r'\b\w+ity\b',
            r'\b\w+ing\b', r'\b\w+er\b', r'\b\w+or\b'
        ]
        
        verb_patterns = [
            r'\b\w+ed\b', r'\b\w+ing\b', r'\bis\b', r'\bare\b', r'\bwas\b', r'\bwere\b',
            r'\bhave\b', r'\bhas\b', r'\bhad\b', r'\bdo\b', r'\bdoes\b', r'\bdid\b'
        ]
        
        adj_patterns = [
            r'\b\w+able\b', r'\b\w+ful\b', r'\b\w+less\b', r'\b\w+ive\b',
            r'\b\w+al\b', r'\b\w+ic\b', r'\b\w+ous\b'
        ]
        
        for idx, row in df.iterrows():
            text = str(row.get('text', '')).lower()
            if not text.strip():
                continue
            
            # Count pattern matches
            noun_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in noun_patterns)
            verb_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in verb_patterns)
            adj_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in adj_patterns)
            
            # Simple heuristics
            words = text.split()
            word_count = len(words)
            
            # Proper nouns (capitalized words)
            propn_count = len([w for w in words if w.istitle() and len(w) > 1])
            
            # Pronouns (common ones)
            pronouns = ['i', 'you', 'he', 'she', 'it', 'we', 'they', 'this', 'that', 'these', 'those']
            pronoun_count = sum(text.count(f' {pronoun} ') for pronoun in pronouns)
            
            # Other POS (remainder)
            total_pos = noun_count + verb_count + adj_count + propn_count + pronoun_count
            other_count = max(0, word_count - total_pos)
            
            # Update dataframe
            df.at[idx, 'num_nouns'] = noun_count
            df.at[idx, 'num_verbs'] = verb_count
            df.at[idx, 'num_adjs'] = adj_count
            df.at[idx, 'num_advs'] = 0  # Hard to detect with simple patterns
            df.at[idx, 'num_propn'] = propn_count
            df.at[idx, 'num_pronouns'] = pronoun_count
            df.at[idx, 'num_other_pos'] = other_count
        
        logger.info("✅ POS features calculated using fallback method")
        return df
    
    def calculate_pos_features(self, df):
        """Main method to calculate POS features"""
        if self.spacy_available:
            return self.calculate_pos_features_spacy(df)
        else:
            return self.calculate_pos_features_fallback(df)
    
    def ensure_pos_features_exist(self, df):
        """Ensure POS feature columns exist with default values"""
        pos_features = ['num_nouns', 'num_verbs', 'num_adjs', 'num_advs', 'num_propn', 'num_pronouns', 'num_other_pos']
        
        missing_features = [feature for feature in pos_features if feature not in df.columns]
        
        if missing_features:
            logger.info(f"🏷️ Adding missing POS features: {missing_features}")
            for feature in missing_features:
                df[feature] = 0
        
        return df

# Global instance
pos_handler = POSFeaturesHandler()

def calculate_pos_features(df):
    """Global function for calculating POS features"""
    return pos_handler.calculate_pos_features(df)

def ensure_pos_features_exist(df):
    """Global function for ensuring POS features exist"""
    return pos_handler.ensure_pos_features_exist(df)

# Test the handler
if __name__ == "__main__":
    print("🏷️ POS FEATURES HANDLER TEST")
    print("=" * 30)
    
    # Create test data
    test_data = {
        'text': [
            'Chapter 1: Introduction',
            'This is a regular sentence with various words.',
            'System Architecture Overview',
            'The implementation uses advanced algorithms.',
            'Conclusion and Future Work'
        ]
    }
    
    df = pd.DataFrame(test_data)
    
    # Test POS features
    handler = POSFeaturesHandler()
    result_df = handler.calculate_pos_features(df)
    
    print("\n📊 POS FEATURES RESULTS:")
    print("-" * 25)
    pos_cols = ['num_nouns', 'num_verbs', 'num_adjs', 'num_advs', 'num_propn', 'num_pronouns', 'num_other_pos']
    for col in pos_cols:
        if col in result_df.columns:
            print(f"{col}: {result_df[col].tolist()}")
    
    print(f"\n✅ spaCy Available: {handler.spacy_available}")
    print("🎉 Test completed successfully!")
