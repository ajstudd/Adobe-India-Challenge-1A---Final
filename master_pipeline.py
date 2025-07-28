#!/usr/bin/env python3
"""
Master Semi-Automated Heading Detection Pipeline
===============================================

ONE SCRIPT TO RULE THEM ALL!

This unified pipeline handles:
✅ Training on labeled data
✅ Processing PDFs to extract blocks  
✅ Generating predictions for manual review
✅ Retraining with corrections
✅ Final JSON output generation
✅ Complete end-to-end workflow

Usage:
    python master_pipeline.py

Author: AI Assistant
Date: July 27, 2025
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
import glob
import json
import logging
import time
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, f1_score, recall_score, precision_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import RFECV
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MasterPipeline:
    """The one and only pipeline you need!"""
    
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load the main configuration (no alternatives)
        self.config = self.load_config()
        
        # Core directories - all data flows through these
        self.labelled_data_dir = os.path.join(self.base_dir, self.config['directories']['labelled_data'])
        self.unprocessed_pdfs_dir = os.path.join(self.base_dir, self.config['directories']['unprocessed_pdfs'])
        self.input_dir = os.path.join(self.base_dir, self.config['directories']['input'])
        self.output_dir = os.path.join(self.base_dir, self.config['directories']['output'])
        
        # Working directories
        self.models_dir = os.path.join(self.base_dir, self.config['directories']['models'])
        self.predictions_dir = os.path.join(self.base_dir, self.config['directories']['predictions'])
        self.reviewed_dir = os.path.join(self.base_dir, self.config['directories']['reviewed'])
        
        # Create all necessary directories
        for dir_path in [self.models_dir, self.predictions_dir, self.reviewed_dir, 
                        self.input_dir, self.output_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Pipeline state
        self.model = None
        self.tfidf_vectorizer = None
        self.feature_columns = None
        self.optimal_threshold = self.config['pipeline_settings']['confidence_threshold']
        self.current_cycle = 0
        
        logger.info("🚀 Master Pipeline initialized!")
        logger.info(f"⚙️  Configuration: {self.config['pipeline_settings']['training_mode']}")
        logger.info(f"📁 Labeled data: {self.labelled_data_dir}")
        logger.info(f"📄 Unprocessed PDFs: {self.unprocessed_pdfs_dir}")
        logger.info(f"📥 Input: {self.input_dir}")
        logger.info(f"📤 Output: {self.output_dir}")
        logger.info("📖 See CONFIG_README.md for configuration options")
    
    def load_config(self):
        """Load configuration from the main config file"""
        config_path = os.path.join(self.base_dir, 'config_main.json')
        
        if not os.path.exists(config_path):
            logger.error(f"❌ Main config file not found: {config_path}")
            logger.info("📖 Please check CONFIG_README.md for configuration options")
            raise FileNotFoundError(f"Main configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"✅ Loaded main configuration: {config_path}")
            logger.info(f"⚙️  Training mode: {config['pipeline_settings']['training_mode']}")
            return config
        except Exception as e:
            logger.error(f"❌ Error loading config: {e}")
            logger.info("📖 Please check CONFIG_README.md for valid configuration format")
            raise
    
    def load_labeled_data(self, min_heading_percentage=None):
        """Load and combine all labeled CSV data"""
        if min_heading_percentage is None:
            min_heading_percentage = self.config['pipeline_settings']['min_heading_percentage']
            
        logger.info("📊 Loading labeled training data...")
        logger.info(f"🎯 Minimum heading percentage: {min_heading_percentage}%")
        
        csv_files = glob.glob(os.path.join(self.labelled_data_dir, "*.csv"))
        csv_files = [f for f in csv_files if not f.endswith('_with_predictions.csv')]
        
        if not csv_files:
            logger.error("❌ No labeled CSV files found!")
            return None
        
        logger.info(f"📄 Found {len(csv_files)} labeled CSV files")
        
        dataframes = []
        total_files = 0
        skipped_files = 0
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                
                if 'is_heading' not in df.columns:
                    logger.warning(f"⚠️  Skipping {os.path.basename(csv_file)} - no 'is_heading' column")
                    skipped_files += 1
                    continue
                
                # Filter out files with very few headings
                heading_percentage = (df['is_heading'].sum() / len(df)) * 100
                if heading_percentage < min_heading_percentage:
                    logger.info(f"⚠️  Skipping {os.path.basename(csv_file)} - only {heading_percentage:.1f}% headings (threshold: {min_heading_percentage}%)")
                    skipped_files += 1
                    continue
                
                df['source_file'] = os.path.basename(csv_file)
                dataframes.append(df)
                total_files += 1
                
                logger.info(f"✅ {os.path.basename(csv_file)}: {len(df)} blocks, {df['is_heading'].sum()} headings ({heading_percentage:.1f}%)")
                
            except Exception as e:
                logger.error(f"❌ Error loading {csv_file}: {e}")
                skipped_files += 1
        
        if not dataframes:
            logger.error("❌ No valid labeled data files found!")
            return None
        
        combined_df = pd.concat(dataframes, ignore_index=True)
        
        logger.info(f"📊 Combined dataset: {len(combined_df)} total blocks")
        logger.info(f"✅ Used {total_files} files, skipped {skipped_files} files")
        logger.info(f"🎯 Total headings: {combined_df['is_heading'].sum()} ({(combined_df['is_heading'].sum()/len(combined_df)*100):.1f}%)")
        
        return combined_df
    
    
    def validate_feature_consistency(self, X):
        """Validate that features match what the model expects"""
        if self.feature_columns is None:
            logger.error("❌ No feature columns defined - model may not be properly loaded")
            return False
        
        expected_features = len(self.feature_columns)
        actual_features = len(X.columns)
        
        logger.info(f"🔍 Feature validation:")
        logger.info(f"   Expected features: {expected_features}")
        logger.info(f"   Actual features:   {actual_features}")
        
        if expected_features != actual_features:
            logger.warning(f"⚠️  Feature count mismatch detected!")
            logger.warning(f"   Expected: {expected_features}, Got: {actual_features}")
            
            # Show TF-IDF info for debugging
            tfidf_cols = [c for c in X.columns if c.startswith('tfidf_')]
            expected_tfidf = [c for c in self.feature_columns if c.startswith('tfidf_')]
            logger.info(f"   TF-IDF features - Expected: {len(expected_tfidf)}, Actual: {len(tfidf_cols)}")
            
            # Show missing features
            missing = set(self.feature_columns) - set(X.columns)
            if missing:
                logger.warning(f"   Missing features ({len(missing)}): {list(missing)[:10]}...")
            
            # Show extra features
            extra = set(X.columns) - set(self.feature_columns)
            if extra:
                logger.warning(f"   Extra features ({len(extra)}): {list(extra)[:10]}...")
            
            # Always attempt alignment for mismatched features
            logger.info("🔧 Will perform automatic feature alignment...")
            return True
        
        logger.info("✅ Feature count matches - no alignment needed")
        return True

    def _add_pos_features(self, df, features):
        """Add Part-of-Speech based features for enhanced heading detection"""
        try:
            import spacy
            from langdetect import detect
            
            # Try to load spaCy model
            try:
                nlp = spacy.load('en_core_web_sm')
            except OSError:
                logger.warning("⚠️  spaCy model not found, skipping POS features")
                # Add zero-filled POS features as fallback
                pos_features = ['num_nouns', 'num_verbs', 'num_adjs', 'num_advs', 'num_propn', 'num_pronouns', 'num_other_pos']
                for feature in pos_features:
                    features[feature] = 0
                return features
            
            logger.info("🔤 Computing POS-based features...")
            
            # Initialize POS feature columns
            features['num_nouns'] = 0
            features['num_verbs'] = 0
            features['num_adjs'] = 0
            features['num_advs'] = 0
            features['num_propn'] = 0
            features['num_pronouns'] = 0
            features['num_other_pos'] = 0
            
            # Process text for POS features (only for English text)
            for idx, text in enumerate(df['text'].fillna('')):
                try:
                    # Quick language detection
                    lang = detect(text) if len(text.strip()) > 3 else 'en'
                    
                    if lang == 'en' and len(text.strip()) > 0:
                        doc = nlp(text)
                        
                        # Count POS tags
                        features.loc[idx, 'num_nouns'] = sum(1 for token in doc if token.pos_ == 'NOUN')
                        features.loc[idx, 'num_verbs'] = sum(1 for token in doc if token.pos_ == 'VERB')
                        features.loc[idx, 'num_adjs'] = sum(1 for token in doc if token.pos_ == 'ADJ')
                        features.loc[idx, 'num_advs'] = sum(1 for token in doc if token.pos_ == 'ADV')
                        features.loc[idx, 'num_propn'] = sum(1 for token in doc if token.pos_ == 'PROPN')
                        features.loc[idx, 'num_pronouns'] = sum(1 for token in doc if token.pos_ == 'PRON')
                        
                        total_pos = len([t for t in doc if t.pos_ != 'SPACE'])
                        counted_pos = (features.loc[idx, 'num_nouns'] + features.loc[idx, 'num_verbs'] + 
                                     features.loc[idx, 'num_adjs'] + features.loc[idx, 'num_advs'] + 
                                     features.loc[idx, 'num_propn'] + features.loc[idx, 'num_pronouns'])
                        features.loc[idx, 'num_other_pos'] = max(0, total_pos - counted_pos)
                        
                except Exception:
                    # Skip problematic text, keep zeros
                    pass
                    
            logger.info(f"   ✅ POS features computed for {len(df)} text blocks")
            return features
            
        except ImportError:
            logger.warning("⚠️  spaCy/langdetect not available, skipping POS features")
            # Add zero-filled POS features as fallback
            pos_features = ['num_nouns', 'num_verbs', 'num_adjs', 'num_advs', 'num_propn', 'num_pronouns', 'num_other_pos']
            for feature in pos_features:
                features[feature] = 0
            return features
    
    def prepare_features(self, df, for_training=True):
        """Prepare enhanced features for training or prediction
        
        Args:
            df: DataFrame with text blocks
            for_training: If True, fit TF-IDF vectorizer. If False, only transform using existing vectorizer.
        """
        logger.info(f"🔧 Preparing enhanced features ({'training' if for_training else 'prediction'} mode)...")
        
        # Basic text and position features
        features = {}
        
        # Enhanced text features
        features['text_length'] = df['text'].str.len().fillna(0)
        features['word_count'] = df['text'].str.split().str.len().fillna(0)
        features['char_count'] = df['text'].str.len().fillna(0)
        features['avg_word_length'] = features['char_count'] / (features['word_count'] + 1)
        
        # Text pattern features
        features['text_upper_ratio'] = df['text'].str.count(r'[A-Z]').fillna(0) / (features['text_length'] + 1)
        features['text_digit_ratio'] = df['text'].str.count(r'\d').fillna(0) / (features['text_length'] + 1)
        features['text_punct_ratio'] = df['text'].str.count(r'[.!?]').fillna(0) / (features['text_length'] + 1)
        features['text_space_ratio'] = df['text'].str.count(r'\s').fillna(0) / (features['text_length'] + 1)
        
        # Heading pattern indicators
        features['starts_with_capital'] = df['text'].str.match(r'^[A-Z]', na=False).astype(int)
        features['all_caps'] = df['text'].str.isupper().fillna(False).astype(int)
        features['title_case'] = df['text'].str.istitle().fillna(False).astype(int)
        features['starts_with_number'] = df['text'].str.match(r'^\d', na=False).astype(int)
        features['ends_with_colon'] = df['text'].str.endswith(':', na=False).astype(int)
        features['has_bold_markers'] = df['text'].str.contains(r'\*\*|\bBOLD\b', na=False).astype(int)
        
        # Word pattern features
        single_word_mask = (features['word_count'] == 1)
        if hasattr(single_word_mask, 'astype'):
            features['single_word'] = single_word_mask.astype(int)
        else:
            features['single_word'] = int(single_word_mask)
            
        short_heading_mask = (features['word_count'].between(2, 5))
        if hasattr(short_heading_mask, 'astype'):
            features['short_heading'] = short_heading_mask.astype(int)
        else:
            features['short_heading'] = int(short_heading_mask)
            
        medium_heading_mask = (features['word_count'].between(6, 10))
        if hasattr(medium_heading_mask, 'astype'):
            features['medium_heading'] = medium_heading_mask.astype(int)
        else:
            features['medium_heading'] = int(medium_heading_mask)
        features['has_common_heading_words'] = df['text'].str.contains(
            r'\b(?:chapter|section|part|introduction|conclusion|summary|abstract|references|appendix)\b', 
            case=False, na=False, regex=True
        ).astype(int)
        
        # Enhanced position features
        if 'x0' in df.columns and 'y0' in df.columns and not (df['x0'] == 0).all():
            features['x_position'] = df['x0'].fillna(df['x0'].median())
            features['y_position'] = df['y0'].fillna(df['y0'].median())
            
            # Additional bounding box features
            if 'x1' in df.columns and 'y1' in df.columns:
                features['width'] = (df['x1'].fillna(0) - df['x0'].fillna(0)).clip(lower=0)
                features['height'] = (df['y1'].fillna(0) - df['y0'].fillna(0)).clip(lower=0)
                features['center_x'] = (df['x0'].fillna(0) + df['x1'].fillna(0)) / 2
                features['center_y'] = (df['y0'].fillna(0) + df['y1'].fillna(0)) / 2
                features['aspect_ratio'] = features['width'] / (features['height'] + 1)
                features['area'] = features['width'] * features['height']
            else:
                features['width'] = 100.0
                features['height'] = 20.0
                features['center_x'] = features['x_position']
                features['center_y'] = features['y_position']
                features['aspect_ratio'] = 5.0
                features['area'] = 2000.0
            
            # Relative position features
            features['x_position_norm'] = (features['x_position'] - features['x_position'].min()) / (features['x_position'].max() - features['x_position'].min() + 1)
            features['y_position_norm'] = (features['y_position'] - features['y_position'].min()) / (features['y_position'].max() - features['y_position'].min() + 1)
            
            # Position-based heading indicators
            try:
                left_aligned_mask = (features['x_position'] <= features['x_position'].quantile(0.1))
                if hasattr(left_aligned_mask, 'astype'):
                    features['left_aligned'] = left_aligned_mask.astype(int)
                else:
                    features['left_aligned'] = int(left_aligned_mask)
                    
                top_of_page_mask = (features['y_position'] <= features['y_position'].quantile(0.1))
                if hasattr(top_of_page_mask, 'astype'):
                    features['top_of_page'] = top_of_page_mask.astype(int)
                else:
                    features['top_of_page'] = int(top_of_page_mask)
                    
                center_mask = (features['center_x'] >= features['center_x'].quantile(0.4)) & (features['center_x'] <= features['center_x'].quantile(0.6))
                # Handle case where center_mask might be a scalar boolean for single samples
                if hasattr(center_mask, 'astype'):
                    features['center_aligned'] = center_mask.astype(int)
                else:
                    # For scalar boolean, convert to integer
                    features['center_aligned'] = int(center_mask)
            except Exception as e:
                # Fallback for edge cases (like single sample testing)
                features['left_aligned'] = 0
                features['top_of_page'] = 0
                features['center_aligned'] = 0
            
        elif 'x' in df.columns and 'y' in df.columns and not (df['x'] == 0).all():
            # Fallback to legacy x,y columns
            features['x_position'] = df['x'].fillna(df['x'].median())
            features['y_position'] = df['y'].fillna(df['y'].median())
            features['width'] = 100.0
            features['height'] = 20.0
            features['center_x'] = features['x_position']
            features['center_y'] = features['y_position']
            features['aspect_ratio'] = 5.0
            features['area'] = 2000.0
            
            # Relative position features
            features['x_position_norm'] = (features['x_position'] - features['x_position'].min()) / (features['x_position'].max() - features['x_position'].min() + 1)
            features['y_position_norm'] = (features['y_position'] - features['y_position'].min()) / (features['y_position'].max() - features['y_position'].min() + 1)
            
            # Position-based heading indicators
            try:
                left_aligned_mask = (features['x_position'] <= features['x_position'].quantile(0.1))
                if hasattr(left_aligned_mask, 'astype'):
                    features['left_aligned'] = left_aligned_mask.astype(int)
                else:
                    features['left_aligned'] = int(left_aligned_mask)
                    
                top_of_page_mask = (features['y_position'] <= features['y_position'].quantile(0.1))
                if hasattr(top_of_page_mask, 'astype'):
                    features['top_of_page'] = top_of_page_mask.astype(int)
                else:
                    features['top_of_page'] = int(top_of_page_mask)
                    
                center_mask = (features['center_x'] >= features['center_x'].quantile(0.4)) & (features['center_x'] <= features['center_x'].quantile(0.6))
                # Handle case where center_mask might be a scalar boolean for single samples
                if hasattr(center_mask, 'astype'):
                    features['center_aligned'] = center_mask.astype(int)
                else:
                    # For scalar boolean, convert to integer
                    features['center_aligned'] = int(center_mask)
            except Exception as e:
                # Fallback for edge cases (like single sample testing)
                features['left_aligned'] = 0
                features['top_of_page'] = 0
                features['center_aligned'] = 0
            
        else:
            logger.warning("⚠️  Position data missing or all zeros - using text-based features only")
            features['x_position'] = 0
            features['y_position'] = 0
            features['width'] = 100.0
            features['height'] = 20.0
            features['center_x'] = 0
            features['center_y'] = 0
            features['aspect_ratio'] = 5.0
            features['area'] = 2000.0
            features['x_position_norm'] = 0
            features['y_position_norm'] = 0
            features['left_aligned'] = 0
            features['top_of_page'] = 0
            features['center_aligned'] = 0
        
        # Enhanced font features with more percentiles
        if 'font_size' in df.columns:
            features['font_size'] = df['font_size'].fillna(df['font_size'].median())
            
            # More granular font size percentiles
            font_percentiles = self.config.get('feature_engineering', {}).get('font_percentiles', [50, 75, 90, 95])
            
            try:
                for percentile in font_percentiles:
                    font_threshold = df['font_size'].quantile(percentile / 100.0)
                    comparison_result = (features['font_size'] >= font_threshold)
                    if hasattr(comparison_result, 'astype'):
                        features[f'font_ge_{percentile}p'] = comparison_result.astype(int)
                    else:
                        # For scalar boolean, convert to integer
                        features[f'font_ge_{percentile}p'] = int(comparison_result)
                
                # Font size statistics
                features['font_size_normalized'] = (features['font_size'] - features['font_size'].min()) / (features['font_size'].max() - features['font_size'].min() + 1)
                features['font_size_zscore'] = (features['font_size'] - features['font_size'].mean()) / (features['font_size'].std() + 1)
                
                largest_font_mask = (features['font_size'] == features['font_size'].max())
                if hasattr(largest_font_mask, 'astype'):
                    features['is_largest_font'] = largest_font_mask.astype(int)
                else:
                    # For scalar boolean, convert to integer
                    features['is_largest_font'] = int(largest_font_mask)
                
                above_median_mask = (features['font_size'] > features['font_size'].median())
                if hasattr(above_median_mask, 'astype'):
                    features['above_median_font'] = above_median_mask.astype(int)
                else:
                    # For scalar boolean, convert to integer
                    features['above_median_font'] = int(above_median_mask)
                    
            except Exception as e:
                logger.warning(f"⚠️  Error in font feature calculation: {e}")
                # Provide fallback values
                for percentile in font_percentiles:
                    features[f'font_ge_{percentile}p'] = 0
                features['font_size_normalized'] = 0.5
                features['font_size_zscore'] = 0
                features['is_largest_font'] = 0
                features['above_median_font'] = 0
            
        else:
            logger.warning("⚠️  Font size data missing")
            features['font_size'] = 12.0
            for p in [50, 75, 90, 95]:
                features[f'font_ge_{p}p'] = 0
            features['font_size_normalized'] = 0.5
            features['font_size_zscore'] = 0
            features['is_largest_font'] = 0
            features['above_median_font'] = 0

        # Additional width/height features derived from position data
        if 'width' not in features:  # Only add if not already calculated from position data
            if 'width' in df.columns and 'height' in df.columns:
                features['width'] = df['width'].fillna(100)
                features['height'] = df['height'].fillna(20)
                features['aspect_ratio'] = features['width'] / (features['height'] + 1)
                features['area'] = features['width'] * features['height']
            else:
                # Use defaults if no width/height data available
                features['width'] = 100.0
                features['height'] = 20.0
                features['aspect_ratio'] = 5.0
                features['area'] = 2000.0
        
        # Shape-based features
        wide_mask = (features['aspect_ratio'] > 5)
        if hasattr(wide_mask, 'astype'):
            features['is_wide'] = wide_mask.astype(int)
        else:
            features['is_wide'] = int(wide_mask)
            
        tall_mask = (features['aspect_ratio'] < 2)
        if hasattr(tall_mask, 'astype'):
            features['is_tall'] = tall_mask.astype(int)
        else:
            features['is_tall'] = int(tall_mask)
        
        # Page-based features
        if 'page_num' in df.columns:
            features['page_num'] = df['page_num'].fillna(1)
            
            first_page_mask = (features['page_num'] == 1)
            if hasattr(first_page_mask, 'astype'):
                features['is_first_page'] = first_page_mask.astype(int)
            else:
                features['is_first_page'] = int(first_page_mask)
                
            early_page_mask = (features['page_num'] <= 3)
            if hasattr(early_page_mask, 'astype'):
                features['is_early_page'] = early_page_mask.astype(int)
            else:
                features['is_early_page'] = int(early_page_mask)
        elif 'page' in df.columns:
            features['page_num'] = df['page'].fillna(1)
            
            first_page_mask = (features['page_num'] == 1)
            if hasattr(first_page_mask, 'astype'):
                features['is_first_page'] = first_page_mask.astype(int)
            else:
                features['is_first_page'] = int(first_page_mask)
                
            early_page_mask = (features['page_num'] <= 3)
            if hasattr(early_page_mask, 'astype'):
                features['is_early_page'] = early_page_mask.astype(int)
            else:
                features['is_early_page'] = int(early_page_mask)
        else:
            features['page_num'] = 1
            features['is_first_page'] = 1
            features['is_early_page'] = 1
        
        # Enhanced TF-IDF features
        logger.info("📝 Computing enhanced TF-IDF features...")
        tfidf_config = self.config['tfidf_params']
        
        if self.tfidf_vectorizer is None:
            # Check if we should be using a loaded vectorizer instead
            if self.model is not None:
                logger.error("❌ TF-IDF vectorizer is None but model is loaded! This is a critical feature mismatch.")
                logger.error("   The model was trained with a different TF-IDF configuration.")
                logger.error("   Please retrain the model or load the correct vectorizer.")
                raise ValueError("Feature mismatch: TF-IDF vectorizer missing for loaded model")
            
            # Only create new vectorizer if we're training (no model loaded yet)
            logger.info("🔧 Creating new TF-IDF vectorizer for training...")
            
            # Adaptive configuration for small datasets
            dataset_size = len(df)
            if dataset_size < 100:
                # Very small dataset - use minimal constraints
                min_df = 1
                max_df = 1.0
                max_features = min(20, dataset_size)
                logger.info("🔧 Using adaptive TF-IDF for small dataset")
            elif dataset_size < 1000:
                # Small dataset - relaxed constraints
                min_df = 1
                max_df = 0.95
                max_features = min(tfidf_config['max_features'], dataset_size // 2)
            else:
                # Normal dataset - use config values
                min_df = tfidf_config['min_df']
                max_df = tfidf_config['max_df']
                max_features = tfidf_config['max_features']
            
            # CRITICAL FIX: Ensure ngram_range is a tuple
            ngram_range = tfidf_config['ngram_range']
            if isinstance(ngram_range, list):
                ngram_range = tuple(ngram_range)
                logger.info(f"🔧 Converted ngram_range from list to tuple: {ngram_range}")
            
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                min_df=min_df,
                max_df=max_df,
                stop_words=tfidf_config['stop_words'],
                sublinear_tf=tfidf_config.get('sublinear_tf', True),
                norm=tfidf_config.get('norm', 'l2')
            )
            
            try:
                tfidf_matrix = self.tfidf_vectorizer.fit_transform(df['text'].fillna(''))
                logger.info(f"✅ TF-IDF fitted with {tfidf_matrix.shape[1]} features")
            except ValueError as e:
                if "no terms remain" in str(e):
                    logger.warning("⚠️  TF-IDF failed, using minimal configuration")
                    # Fallback to most minimal settings
                    self.tfidf_vectorizer = TfidfVectorizer(
                        max_features=min(10, dataset_size),
                        ngram_range=(1, 1),
                        min_df=1,
                        max_df=1.0,
                        stop_words=None
                    )
                    tfidf_matrix = self.tfidf_vectorizer.fit_transform(df['text'].fillna(''))
                    logger.info(f"✅ TF-IDF fitted with fallback config: {tfidf_matrix.shape[1]} features")
                else:
                    raise e
        else:
            # Use existing vectorizer for prediction (transform only, don't fit)
            logger.info("🔧 Using existing TF-IDF vectorizer for prediction...")
            tfidf_matrix = self.tfidf_vectorizer.transform(df['text'].fillna(''))
            logger.info(f"✅ TF-IDF transformed to {tfidf_matrix.shape[1]} features")
        
        # Add TF-IDF features
        tfidf_feature_names = [f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])]
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_feature_names)
        
        # Convert features dict to DataFrame for POS processing
        features_df = pd.DataFrame(features)
        
        # Add POS-based features if enabled
        pos_features_enabled = self.config.get('feature_engineering', {}).get('pos_features', False)
        if pos_features_enabled:
            features_df = self._add_pos_features(df, features_df)
        else:
            # Add zero-filled POS features for consistency
            pos_features = ['num_nouns', 'num_verbs', 'num_adjs', 'num_advs', 'num_propn', 'num_pronouns', 'num_other_pos']
            for feature in pos_features:
                features_df[feature] = 0
        
        # Add derived POS features (regardless of whether POS is enabled)
        logger.info("🏷️  Adding derived POS features...")
        pos_columns = ['num_nouns', 'num_verbs', 'num_adjs', 'num_advs', 'num_propn', 'num_pronouns', 'num_other_pos']
        
        # Calculate total POS tags
        features_df['total_pos_tags'] = features_df[pos_columns].sum(axis=1)
        
        # Calculate POS ratios (with safe division)
        features_df['noun_ratio'] = features_df['num_nouns'] / (features_df['total_pos_tags'] + 1)
        features_df['verb_ratio'] = features_df['num_verbs'] / (features_df['total_pos_tags'] + 1)
        features_df['adj_ratio'] = features_df['num_adjs'] / (features_df['total_pos_tags'] + 1)
        features_df['propn_ratio'] = features_df['num_propn'] / (features_df['total_pos_tags'] + 1)
        
        # Common heading POS patterns
        features_df['high_noun_content'] = (features_df['noun_ratio'] > 0.5).astype(int)
        features_df['has_proper_nouns'] = (features_df['num_propn'] > 0).astype(int)
        features_df['minimal_verbs'] = (features_df['verb_ratio'] < 0.2).astype(int)
        pos_pattern_mask = (
            features_df['high_noun_content'] & 
            features_df['minimal_verbs']
        )
        features_df['heading_pos_pattern'] = pos_pattern_mask.astype(int)
        
        # Combine all features
        feature_df = pd.concat([features_df, tfidf_df], axis=1)
        
        # Store feature columns for later use
        self.feature_columns = feature_df.columns.tolist()
        
        pos_feature_count = len([c for c in self.feature_columns if any(pos in c for pos in ['num_', 'ratio', 'pos_', 'noun', 'verb', 'adj'])])
        logger.info(f"✅ Added {pos_feature_count} total POS and derived features")
        
        logger.info(f"✅ Prepared {len(self.feature_columns)} enhanced features")
        logger.info(f"   📝 Text features: {len([c for c in self.feature_columns if 'text_' in c or 'word_' in c or 'char_' in c])}")
        logger.info(f"   📏 Font features: {len([c for c in self.feature_columns if 'font_' in c])}")
        logger.info(f"   📍 Position features: {len([c for c in self.feature_columns if 'position' in c or 'aligned' in c])}")
        logger.info(f"   🏷️  POS features: {len([c for c in self.feature_columns if 'num_' in c or 'pos' in c or 'ratio' in c])}")
        logger.info(f"   📄 TF-IDF features: {len(tfidf_feature_names)}")
        
        return feature_df
    
    def train_model(self, df, test_size=None):
        """Train the heading detection model with enhanced settings"""
        logger.info("🌳 Training enhanced heading detection model...")
        
        if test_size is None:
            test_size = self.config.get('training', {}).get('test_size', 0.15)
        
        # Prepare enhanced features
        X = self.prepare_features(df)
        y = df['is_heading'].values
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        logger.info(f"📊 Training: {len(X_train)} samples, Testing: {len(X_test)} samples")
        logger.info(f"🎯 Training headings: {y_train.sum()}/{len(y_train)} ({y_train.mean()*100:.1f}%)")
        
        # Apply enhanced SMOTE for better class balance
        logger.info("⚖️  Applying enhanced SMOTE for class balancing...")
        smote_config = self.config['smote_params']
        
        # Check if dataset is too small for SMOTE
        minority_class_count = min(sum(y_train == 0), sum(y_train == 1))
        required_neighbors = smote_config.get('k_neighbors', 5)
        
        if len(X_train) < 10 or minority_class_count <= required_neighbors:
            logger.warning(f"⚠️  Dataset too small for SMOTE ({len(X_train)} samples, {minority_class_count} minority samples)")
            logger.warning(f"    Skipping SMOTE and using original training data")
            X_train_balanced = X_train
            y_train_balanced = y_train
        else:
            # Adjust sampling strategy for small datasets
            original_strategy = smote_config['sampling_strategy']
            
            # For small datasets, use a more conservative sampling strategy
            if len(X_train) < 100:
                adjusted_strategy = min(original_strategy, 0.8)  # Cap at 0.8 for small datasets
                logger.info(f"   📉 Adjusting SMOTE strategy from {original_strategy} to {adjusted_strategy} for small dataset")
            else:
                adjusted_strategy = original_strategy
            
            # Also adjust k_neighbors for small datasets
            adjusted_k_neighbors = min(required_neighbors, minority_class_count - 1)
            if adjusted_k_neighbors < 1:
                adjusted_k_neighbors = 1
            
            smote = SMOTE(
                sampling_strategy=adjusted_strategy, 
                random_state=smote_config['random_state'],
                k_neighbors=adjusted_k_neighbors
            )
            
            try:
                X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
                logger.info(f"✅ SMOTE successful with k_neighbors={adjusted_k_neighbors}")
            except ValueError as e:
                logger.warning(f"⚠️  SMOTE failed: {e}")
                logger.warning(f"    Using original training data without SMOTE")
                X_train_balanced = X_train
                y_train_balanced = y_train
        
        logger.info(f"📈 After balancing: {len(X_train_balanced)} samples, {y_train_balanced.sum()} headings ({y_train_balanced.mean()*100:.1f}%)")
        
        # Get model configuration early for use in RFECV
        model_config = self.config['model_params']
        
        # Optional RFECV Feature Selection
        feature_selection_enabled = self.config.get('training', {}).get('feature_selection', False)
        if feature_selection_enabled and len(self.feature_columns) > 100:
            logger.info("🔍 Applying RFECV feature selection...")
            
            # Use a simpler model for feature selection to save time
            rf_selector = RandomForestClassifier(
                n_estimators=50,  # Fewer trees for speed
                max_depth=10,
                random_state=42,
                n_jobs=-1,
                class_weight={int(k): v for k, v in model_config['class_weight'].items()}
            )
            
            # Use RFECV with cross-validation
            rfecv = RFECV(
                estimator=rf_selector,
                step=10,  # Remove 10 features at a time for speed
                cv=3,     # 3-fold CV for speed
                scoring='f1',
                n_jobs=-1,
                verbose=1
            )
            
            logger.info(f"   🔬 Starting with {X_train_balanced.shape[1]} features")
            
            # Fit RFECV
            start_time = time.time()
            X_train_selected = rfecv.fit_transform(X_train_balanced, y_train_balanced)
            X_test_selected = rfecv.transform(X_test)
            selection_time = time.time() - start_time
            
            # Update feature columns to only selected ones
            selected_features = [self.feature_columns[i] for i in range(len(self.feature_columns)) if rfecv.support_[i]]
            self.feature_columns = selected_features
            
            logger.info(f"   ✅ RFECV completed in {selection_time:.1f}s")
            logger.info(f"   🎯 Selected {X_train_selected.shape[1]} optimal features")
            logger.info(f"   📉 Reduced features by {X_train_balanced.shape[1] - X_train_selected.shape[1]}")
            
            # Use selected features for training
            X_train_balanced = X_train_selected
            X_test = X_test_selected
            
        elif feature_selection_enabled:
            logger.info(f"⚠️  Skipping RFECV (only {len(self.feature_columns)} features, threshold: 100)")
        
        # Train Random Forest with enhanced configuration
        logger.info(f"🔧 Enhanced Model Configuration:")
        logger.info(f"   🌳 Trees (n_estimators): {model_config['n_estimators']}")
        logger.info(f"   📏 Max Depth: {model_config['max_depth']}")
        logger.info(f"   🍃 Min Samples Split: {model_config['min_samples_split']}")
        logger.info(f"   🍂 Min Samples Leaf: {model_config['min_samples_leaf']}")
        logger.info(f"   ⚖️  Class Weight: {model_config['class_weight']}")
        logger.info(f"   🎲 Max Features: {model_config.get('max_features', 'sqrt')}")
        logger.info(f"   📦 Max Samples: {model_config.get('max_samples', 'auto')}")
        
        # Calculate expected training time
        expected_time = self._estimate_detailed_training_time(
            model_config['n_estimators'], 
            len(X_train_balanced), 
            X_train_balanced.shape[1]
        )
        logger.info(f"⏱️  Expected training time: {expected_time}")
        
        start_time = time.time()
        
        self.model = RandomForestClassifier(
            n_estimators=model_config['n_estimators'],
            max_depth=model_config['max_depth'],
            min_samples_split=model_config['min_samples_split'],
            min_samples_leaf=model_config['min_samples_leaf'],
            max_features=model_config.get('max_features', 'sqrt'),
            max_samples=model_config.get('max_samples', None),
            bootstrap=model_config.get('bootstrap', True),
            random_state=42,
            n_jobs=-1,  # Use all CPU cores
            class_weight={int(k): v for k, v in model_config['class_weight'].items()},
            verbose=1,  # Show progress
            warm_start=False
        )
        
        logger.info(f"🏋️  Training Random Forest with {model_config['n_estimators']} trees...")
        logger.info(f"🖥️  Using all CPU cores (Ryzen 5 4600H - 6 cores/12 threads)")
        
        # Train with progress tracking
        self.model.fit(X_train_balanced, y_train_balanced)
        
        training_time = time.time() - start_time
        logger.info(f"⏱️  Training completed in {training_time:.1f} seconds ({training_time/60:.1f} minutes)")
        
        # Validate training time expectations
        if model_config['n_estimators'] >= 1000 and training_time < 300:  # Less than 5 minutes
            logger.warning(f"⚠️  Training completed quickly ({training_time:.1f}s) for {model_config['n_estimators']} trees.")
            logger.warning(f"    This might indicate:")
            logger.warning(f"    - Small effective dataset after preprocessing")
            logger.warning(f"    - Simple features that don't require much computation")
            logger.warning(f"    - Consider adding more complex features or data")
        
        # Enhanced prediction and evaluation
        y_test_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Find optimal threshold with more granular search
        self.optimal_threshold = self._find_enhanced_threshold(y_test, y_test_proba)
        
        # Final predictions with optimal threshold
        y_test_pred = (y_test_proba >= self.optimal_threshold).astype(int)
        
        # Calculate comprehensive metrics including ROC AUC
        test_accuracy = (y_test == y_test_pred).mean()
        test_f1 = f1_score(y_test, y_test_pred)
        test_recall = recall_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred)
        
        # Enhanced ROC & AUC evaluation
        try:
            test_auc = roc_auc_score(y_test, y_test_proba)
            fpr, tpr, roc_thresholds = roc_curve(y_test, y_test_proba)
            
            # Find best threshold from ROC curve (Youden's index)
            youden_index = tpr - fpr
            best_roc_threshold_idx = np.argmax(youden_index)
            best_roc_threshold = roc_thresholds[best_roc_threshold_idx]
            
            logger.info(f"📈 ROC AUC Analysis:")
            logger.info(f"   🎯 ROC AUC Score: {test_auc:.3f}")
            logger.info(f"   🔧 Best ROC Threshold (Youden): {best_roc_threshold:.3f}")
            logger.info(f"   📊 TPR at optimal: {tpr[best_roc_threshold_idx]:.3f}")
            logger.info(f"   📊 FPR at optimal: {fpr[best_roc_threshold_idx]:.3f}")
            
        except Exception as e:
            logger.warning(f"⚠️  ROC AUC calculation failed: {e}")
            test_auc = 0.5
        
        logger.info(f"✅ Enhanced model training completed!")
        logger.info(f"📈 Test Accuracy: {test_accuracy:.3f}")
        logger.info(f"🎯 Test F1-Score: {test_f1:.3f}")
        logger.info(f"📊 Test Recall: {test_recall:.3f}")
        logger.info(f"🔍 Test Precision: {test_precision:.3f}")
        logger.info(f"🔧 Optimal Threshold: {self.optimal_threshold:.3f}")
        logger.info(f"📊 Predicted {y_test_pred.sum()}/{len(y_test)} headings in test set")
        
        # Feature importance analysis
        self._log_feature_importance()
        
        return {
            'test_accuracy': test_accuracy,
            'test_f1': test_f1,
            'test_recall': test_recall,
            'test_precision': test_precision,
            'optimal_threshold': self.optimal_threshold,
            'predicted_headings': y_test_pred.sum(),
            'total_test_samples': len(y_test),
            'training_time': training_time
        }
    
    def _estimate_training_time(self, n_estimators, n_samples):
        """Estimate training time based on trees and samples"""
        if n_estimators >= 1000:
            return "45-90 minutes"
        elif n_estimators >= 500:
            return "20-45 minutes"
        elif n_estimators >= 200:
            return "10-20 minutes"
        else:
            return "3-10 minutes"
    
    def _estimate_detailed_training_time(self, n_estimators, n_samples, n_features):
        """Provide detailed training time estimate for Ryzen 5 4600H"""
        # Base computation time per tree (empirical estimate for your CPU)
        base_time_per_tree = 0.5  # seconds per tree for moderate complexity
        
        # Adjust for sample size
        sample_factor = min(2.0, n_samples / 10000)  # More samples = longer training
        
        # Adjust for feature complexity
        feature_factor = min(1.5, n_features / 100)  # More features = longer training
        
        estimated_seconds = n_estimators * base_time_per_tree * sample_factor * feature_factor
        
        if estimated_seconds < 60:
            return f"{estimated_seconds:.0f} seconds"
        elif estimated_seconds < 3600:
            return f"{estimated_seconds/60:.1f} minutes"
        else:
            return f"{estimated_seconds/3600:.1f} hours"
    
    def comprehensive_accuracy_test(self):
        """Run comprehensive accuracy testing with detailed logging"""
        try:
            from comprehensive_accuracy_test import AccuracyLogger
            
            logger.info("🔍 Running comprehensive accuracy test...")
            accuracy_logger = AccuracyLogger()
            success = accuracy_logger.run_full_accuracy_test()
            
            if success:
                logger.info("✅ Comprehensive accuracy test completed successfully!")
                logger.info("📁 Check accuracy_logs/ folder for detailed reports")
                return True
            else:
                logger.error("❌ Comprehensive accuracy test failed!")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error running comprehensive accuracy test: {e}")
            return False
    
    def find_optimal_threshold(self, y_true, y_proba):
        """Find optimal threshold for classification"""
        thresholds = np.arange(0.1, 0.9, 0.02)
        best_threshold = 0.5
        best_f1 = 0
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            f1 = f1_score(y_true, y_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        return best_threshold
    
    def _find_enhanced_threshold(self, y_true, y_proba):
        """Find optimal threshold with granular search focused on recall"""
        thresholds = np.arange(0.1, 0.8, 0.02)  # More granular search
        best_threshold = 0.5
        best_score = 0
        
        # Optimize for balanced F1 with emphasis on recall
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            f1 = f1_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            
            # Combined score favoring recall
            combined_score = 0.4 * f1 + 0.6 * recall
            
            if combined_score > best_score:
                best_score = combined_score
                best_threshold = threshold
        
        return best_threshold
    
    def _log_feature_importance(self):
        """Log top feature importance for analysis"""
        if self.model and hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            feature_names = self.feature_columns
            
            # Get top 15 features
            indices = np.argsort(importance)[::-1][:15]
            
            logger.info(f"🔍 Top 15 Most Important Features:")
            for i, idx in enumerate(indices):
                logger.info(f"   {i+1:2d}. {feature_names[idx]:<25} {importance[idx]:.4f}")
    
    def save_model(self, version="latest"):
        """Save the trained model"""
        if self.model is None:
            logger.error("❌ No model to save")
            return None
        
        if self.tfidf_vectorizer is None:
            logger.error("❌ No TF-IDF vectorizer to save")
            return None
        
        if self.feature_columns is None:
            logger.error("❌ No feature columns to save")
            return None
        
        model_path = os.path.join(self.models_dir, f"heading_model_{version}.pkl")
        vectorizer_path = os.path.join(self.models_dir, f"tfidf_vectorizer_{version}.pkl")
        
        try:
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'optimal_threshold': self.optimal_threshold,
                    'feature_columns': self.feature_columns
                }, f)
            
            with open(vectorizer_path, 'wb') as f:
                pickle.dump(self.tfidf_vectorizer, f)
            
            logger.info(f"💾 Model saved: {model_path}")
            logger.info(f"💾 TF-IDF vectorizer saved: {vectorizer_path}")
            logger.info(f"🔧 Saved {len(self.feature_columns)} feature columns")
            return model_path
        except Exception as e:
            logger.error(f"❌ Error saving model: {e}")
            return None
    
    def load_model(self, version="latest"):
        """Load a saved model"""
        model_path = os.path.join(self.models_dir, f"heading_model_{version}.pkl")
        vectorizer_path = os.path.join(self.models_dir, f"tfidf_vectorizer_{version}.pkl")
        
        if not os.path.exists(model_path):
            logger.warning(f"⚠️  Model not found: {model_path}")
            return False
        
        if not os.path.exists(vectorizer_path):
            logger.warning(f"⚠️  TF-IDF vectorizer not found: {vectorizer_path}")
            logger.warning("   This will cause feature mismatch errors during prediction.")
            return False
        
        try:
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.optimal_threshold = data['optimal_threshold']
                self.feature_columns = data['feature_columns']
            
            with open(vectorizer_path, 'rb') as f:
                self.tfidf_vectorizer = pickle.load(f)
            
            logger.info(f"✅ Model loaded: {model_path}")
            logger.info(f"✅ TF-IDF vectorizer loaded: {vectorizer_path}")
            logger.info(f"🔧 Expected features: {len(self.feature_columns)}")
            logger.info(f"🔧 Optimal threshold: {self.optimal_threshold}")
            return True
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            # Reset everything if loading fails
            self.model = None
            self.tfidf_vectorizer = None
            self.feature_columns = None
            return False
    
    def extract_pdf_to_blocks(self, pdf_path):
        """Extract blocks from PDF using available methods"""
        logger.info(f"📄 Extracting blocks from {os.path.basename(pdf_path)}...")
        
        # Try to use existing extraction scripts
        try:
            sys.path.insert(0, os.path.join(self.base_dir, 'src', 'extraction'))
            from extract_local_dataset_to_csv import process_pdf
            
            output_csv = os.path.join(self.predictions_dir, f"{Path(pdf_path).stem}_extracted.csv")
            result = process_pdf(pdf_path, output_csv)
            
            if os.path.exists(output_csv):
                df = pd.read_csv(output_csv)
                logger.info(f"✅ Extracted {len(df)} blocks using extract_local_dataset_to_csv")
                return df
        except Exception as e:
            logger.warning(f"⚠️  extract_local_dataset_to_csv failed: {e}")
        
        # Fallback to basic PyMuPDF extraction
        try:
            import fitz  # PyMuPDF
            
            doc = fitz.open(pdf_path)
            blocks = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                block_dict = page.get_text("dict")
                
                for block in block_dict["blocks"]:
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                blocks.append({
                                    'text': span['text'].strip(),
                                    'x': span['bbox'][0],
                                    'y': span['bbox'][1],
                                    'width': span['bbox'][2] - span['bbox'][0],
                                    'height': span['bbox'][3] - span['bbox'][1],
                                    'font_size': span['size'],
                                    'page_num': page_num + 1,
                                    'is_heading': 0  # To be predicted
                                })
            
            doc.close()
            df = pd.DataFrame(blocks)
            
            # Filter out empty or very short text
            df = df[df['text'].str.len() > 3].reset_index(drop=True)
            
            logger.info(f"✅ Extracted {len(df)} blocks using PyMuPDF fallback")
            return df
            
        except Exception as e:
            logger.error(f"❌ PyMuPDF extraction failed: {e}")
            return None
    
    def predict_headings(self, df):
        """Predict headings for a DataFrame of blocks"""
        if self.model is None:
            logger.error("❌ No model loaded. Train or load a model first.")
            return None
        
        if self.tfidf_vectorizer is None:
            logger.error("❌ TF-IDF vectorizer not loaded. Cannot proceed with prediction.")
            return None

        # Prepare features (using existing vectorizer)
        X = self.prepare_features(df)
        
        # Validate features before alignment
        if not self.validate_feature_consistency(X):
            logger.error("❌ Feature validation failed. Cannot proceed with prediction.")
            logger.error("   This usually means the model was trained with a different configuration.")
            logger.error("   Please retrain the model or use the correct configuration.")
            return None
        
        # Enhanced feature alignment - always align to ensure consistency
        logger.info(f"🔧 Aligning features: {len(X.columns)} → {len(self.feature_columns)}")
        
        # Create a new DataFrame with only the expected features in the correct order
        aligned_X = pd.DataFrame(index=X.index)
        missing_count = 0
        
        for col in self.feature_columns:
            if col in X.columns:
                aligned_X[col] = X[col]
            else:
                # Use appropriate default values based on feature type
                if 'tfidf_' in col:
                    aligned_X[col] = 0.0
                elif any(keyword in col.lower() for keyword in ['ratio', 'norm', 'relative']):
                    aligned_X[col] = 0.0
                elif any(keyword in col.lower() for keyword in ['count', 'length', 'size', 'position']):
                    aligned_X[col] = 0.0
                elif any(keyword in col.lower() for keyword in ['is_', 'has_', 'starts_', 'ends_']):
                    aligned_X[col] = 0
                else:
                    aligned_X[col] = 0.0
                missing_count += 1
        
        if missing_count > 0:
            logger.warning(f"   🔧 Filled {missing_count} missing features with appropriate defaults")
        
        # Log any extra features that are being dropped
        extra_features = set(X.columns) - set(self.feature_columns)
        if extra_features:
            logger.info(f"   📝 Dropping {len(extra_features)} extra features from this prediction")
        
        X = aligned_X
        logger.info(f"✅ Features aligned: {len(X.columns)} features in model order")
        
        # Final validation
        if len(X.columns) != len(self.feature_columns):
            logger.error(f"❌ Feature alignment failed! Still have {len(X.columns)} instead of {len(self.feature_columns)}")
            return None
        
        # Predict probabilities
        probabilities = self.model.predict_proba(X)[:, 1]
        predictions = (probabilities >= self.optimal_threshold).astype(int)
        
        # Add predictions to dataframe
        result_df = df.copy()
        result_df['is_heading'] = predictions
        result_df['heading_probability'] = probabilities
        result_df['predicted_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        return result_df
    
    def process_pdf_batch(self, max_pdfs=None):
        """Process all PDFs in unprocessed_pdfs directory"""
        logger.info("\\n" + "="*70)
        logger.info("📄 PROCESSING PDF BATCH FOR PREDICTIONS")
        logger.info("="*70)
        
        if self.model is None:
            logger.error("❌ No model loaded. Train a model first.")
            return False
        
        pdf_files = list(Path(self.unprocessed_pdfs_dir).glob("*.pdf"))
        
        if not pdf_files:
            logger.warning("⚠️  No PDF files found in unprocessed_pdfs directory")
            return True
        
        if max_pdfs:
            pdf_files = pdf_files[:max_pdfs]
        
        logger.info(f"📄 Found {len(pdf_files)} PDF files to process")
        
        successful = 0
        failed = 0
        
        for i, pdf_path in enumerate(pdf_files):
            pdf_name = pdf_path.stem
            logger.info(f"\\n--- Processing PDF {i+1}/{len(pdf_files)}: {pdf_name} ---")
            
            try:
                # Extract blocks
                df_blocks = self.extract_pdf_to_blocks(str(pdf_path))
                
                if df_blocks is None or len(df_blocks) == 0:
                    logger.error(f"❌ Failed to extract blocks from {pdf_name}")
                    failed += 1
                    continue
                
                # Generate predictions
                df_predictions = self.predict_headings(df_blocks)
                
                if df_predictions is None:
                    logger.error(f"❌ Failed to generate predictions for {pdf_name}")
                    failed += 1
                    continue
                
                # Save predictions for review
                predictions_path = os.path.join(self.predictions_dir, f"{pdf_name}_predictions.csv")
                df_predictions.to_csv(predictions_path, index=False)
                
                predicted_headings = df_predictions['is_heading'].sum()
                logger.info(f"✅ Predicted {predicted_headings} headings out of {len(df_predictions)} blocks")
                logger.info(f"💾 Saved predictions: {predictions_path}")
                
                successful += 1
                
            except Exception as e:
                logger.error(f"❌ Error processing {pdf_name}: {e}")
                failed += 1
        
        logger.info(f"\\n🎯 PDF processing complete!")
        logger.info(f"✅ Successfully processed: {successful} PDFs")
        logger.info(f"❌ Failed: {failed} PDFs")
        logger.info(f"📁 Predictions saved to: {self.predictions_dir}")
        
        return successful > 0
    
    def wait_for_review(self, check_interval_minutes=5):
        """Wait for user to review and correct predictions"""
        logger.info("\\n" + "="*70)
        logger.info("👁️  WAITING FOR MANUAL REVIEW")
        logger.info("="*70)
        
        logger.info(f"📁 Please review predictions in: {self.predictions_dir}")
        logger.info("📝 Instructions:")
        logger.info("   1. Open CSV files in the predictions folder")
        logger.info("   2. Correct the 'is_heading' column (0 or 1)")
        logger.info("   3. Save corrected files to the 'reviewed' folder")
        logger.info("   4. The pipeline will automatically detect reviewed files")
        
        while True:
            reviewed_files = list(Path(self.reviewed_dir).glob("*_predictions.csv"))
            
            if reviewed_files:
                logger.info(f"✅ Found {len(reviewed_files)} reviewed files!")
                break
            
            logger.info(f"⏳ Waiting for reviews... (checking every {check_interval_minutes} min)")
            time.sleep(check_interval_minutes * 60)
    
    def retrain_with_corrections(self):
        """Retrain model using original data + corrected predictions"""
        logger.info("\\n" + "="*70)
        logger.info("🔄 RETRAINING WITH CORRECTIONS")
        logger.info("="*70)
        
        # Load original labeled data
        original_df = self.load_labeled_data(min_heading_percentage=0.5)
        
        if original_df is None:
            logger.error("❌ Failed to load original labeled data")
            return False
        
        # Load reviewed corrections
        reviewed_files = list(Path(self.reviewed_dir).glob("*_predictions.csv"))
        
        if not reviewed_files:
            logger.warning("⚠️  No reviewed files found for retraining")
            return False
        
        reviewed_dataframes = []
        for file_path in reviewed_files:
            try:
                df = pd.read_csv(file_path)
                if 'is_heading' in df.columns:
                    df['source_file'] = f"reviewed_{os.path.basename(file_path)}"
                    reviewed_dataframes.append(df)
                    logger.info(f"✅ Loaded {os.path.basename(file_path)}: {len(df)} blocks, {df['is_heading'].sum()} headings")
            except Exception as e:
                logger.error(f"❌ Error loading {file_path}: {e}")
        
        if not reviewed_dataframes:
            logger.error("❌ No valid reviewed data found")
            return False
        
        # Combine original + reviewed data
        all_dataframes = [original_df] + reviewed_dataframes
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        
        logger.info(f"📊 Combined training data: {len(combined_df)} blocks")
        logger.info(f"🎯 Total headings: {combined_df['is_heading'].sum()} ({combined_df['is_heading'].mean()*100:.1f}%)")
        
        # Retrain model
        self.current_cycle += 1
        results = self.train_model(combined_df)
        
        # Save updated model
        model_path = self.save_model(f"cycle_{self.current_cycle}")
        
        logger.info(f"✅ Retraining completed! (Cycle {self.current_cycle})")
        return True
    
    def generate_json_output(self, process_input_folder=True):
        """Generate final JSON output for competition submission"""
        logger.info("\\n" + "="*70)
        logger.info("📤 GENERATING JSON OUTPUT")
        logger.info("="*70)
        
        if self.model is None:
            logger.warning("⚠️  No model loaded. Attempting to load existing model...")
            if not self.load_model():
                logger.error("❌ No model found. Please train a model first.")
                return False
        
        if self.tfidf_vectorizer is None:
            logger.warning("⚠️  TF-IDF vectorizer not loaded. Attempting to load...")
            if not self.load_model():
                logger.error("❌ Failed to load model and vectorizer")
                return False
        
        # Check feature compatibility by testing with sample data
        try:
            # Create a small test sample to check feature compatibility
            test_sample = pd.DataFrame({
                'text': ['Test heading sample'],
                'font_size': [12],
                'page_num': [1],
                'x': [0], 'y': [0], 'width': [100], 'height': [20]
            })
            test_features = self.prepare_features(test_sample)
            
            if len(test_features.columns) != len(self.feature_columns):
                logger.error("❌ FEATURE MISMATCH DETECTED!")
                logger.error(f"   Current config generates: {len(test_features.columns)} features")
                logger.error(f"   Loaded model expects: {len(self.feature_columns)} features")
                logger.error("   This means the model was trained with a different configuration.")
                logger.error("")
                logger.error("🔧 SOLUTIONS:")
                logger.error("   1. Retrain the model with current configuration (option 1a in menu)")
                logger.error("   2. Change configuration to match the trained model")
                logger.error("   3. Use the retrain_advanced.py script")
                logger.error("")
                
                # Ask user what to do
                user_choice = input("Would you like to retrain the model now? (y/N): ").strip().lower()
                if user_choice == 'y':
                    logger.info("🔄 Retraining model with current configuration...")
                    df = self.load_labeled_data()
                    if df is not None:
                        self.train_model(df)
                        self.save_model()
                        logger.info("✅ Model retrained successfully!")
                    else:
                        logger.error("❌ Failed to load training data")
                        return False
                else:
                    logger.error("❌ Cannot proceed with feature mismatch. Please retrain the model.")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Error checking feature compatibility: {e}")
            return False

        # Determine which PDFs to process
        if process_input_folder and os.path.exists(self.input_dir):
            pdf_files = list(Path(self.input_dir).glob("*.pdf"))
            source_dir = "input"
        else:
            pdf_files = list(Path(self.unprocessed_pdfs_dir).glob("*.pdf"))
            source_dir = "unprocessed_pdfs"
        
        if not pdf_files:
            logger.warning(f"⚠️  No PDF files found in {source_dir} directory")
            return False
        
        logger.info(f"📄 Processing {len(pdf_files)} PDFs from {source_dir} folder")
        
        successful = 0
        
        for pdf_path in pdf_files:
            pdf_name = pdf_path.stem
            logger.info(f"\\n--- Generating JSON for: {pdf_name} ---")
            
            try:
                # Extract and predict
                df_blocks = self.extract_pdf_to_blocks(str(pdf_path))
                
                if df_blocks is None or len(df_blocks) == 0:
                    logger.error(f"❌ Failed to extract blocks from {pdf_name}")
                    continue
                
                df_predictions = self.predict_headings(df_blocks)
                
                if df_predictions is None:
                    logger.error(f"❌ Failed to generate predictions for {pdf_name}")
                    continue
                
                # Filter predicted headings
                headings_df = df_predictions[df_predictions['is_heading'] == 1].copy()
                
                if len(headings_df) == 0:
                    logger.warning(f"⚠️  No headings detected in {pdf_name}")
                
                # Create JSON structure
                outline = []
                for _, row in headings_df.iterrows():
                    # Simple heuristic for heading levels based on font size and position
                    font_size = row.get('font_size', 12)
                    y_position = row.get('y_position', 0)
                    
                    if font_size >= 16 or y_position < 100:
                        level = "H1"
                    elif font_size >= 14:
                        level = "H2"
                    else:
                        level = "H3"
                    
                    outline.append({
                        "level": level,
                        "text": str(row['text']).strip(),
                        "page": int(row.get('page_num', 1))
                    })
                
                # Create final JSON
                json_output = {
                    "title": pdf_name.replace('_', ' ').title(),
                    "outline": outline
                }
                
                # Save JSON
                json_path = os.path.join(self.output_dir, f"{pdf_name}.json")
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(json_output, f, indent=2, ensure_ascii=False)
                
                logger.info(f"✅ Generated JSON with {len(outline)} headings")
                logger.info(f"💾 Saved: {json_path}")
                
                successful += 1
                
            except Exception as e:
                logger.error(f"❌ Error processing {pdf_name}: {e}")
        
        logger.info(f"\\n🎯 JSON generation complete!")
        logger.info(f"✅ Successfully processed: {successful}/{len(pdf_files)} PDFs")
        logger.info(f"📁 JSON files saved to: {self.output_dir}")
        
        return successful > 0
    
    def run_full_cycle(self, max_pdfs=None):
        """Run one complete semi-supervised learning cycle"""
        logger.info("\\n" + "="*80)
        logger.info("🚀 RUNNING FULL SEMI-SUPERVISED CYCLE")
        logger.info("="*80)
        
        # Step 1: Train initial model
        if not self.load_model():
            logger.info("🌱 No existing model found, training new model...")
            df = self.load_labeled_data()
            if df is None:
                return False
            
            self.train_model(df)
            self.save_model()
        else:
            logger.info("✅ Loaded existing model")
        
        # Step 2: Process PDFs
        if not self.process_pdf_batch(max_pdfs):
            logger.error("❌ Failed to process PDFs")
            return False
        
        # Step 3: Wait for review (optional - can be skipped)
        logger.info("\\n⏸️  Manual review step:")
        logger.info("   - Review predictions in the 'predictions' folder")
        logger.info("   - Place corrected files in the 'reviewed' folder")
        logger.info("   - Or press Enter to skip and use current model")
        
        user_input = input("\\nWait for manual review? (y/N): ").strip().lower()
        
        if user_input == 'y':
            self.wait_for_review()
            self.retrain_with_corrections()
        
        # Step 4: Generate final JSON output
        self.generate_json_output()
        
        logger.info("\\n🎉 Full cycle completed successfully!")
        return True
    
    def interactive_menu(self):
        """Interactive menu for pipeline operations"""
        while True:
            print("\n" + "="*60)
            print("🤖 MASTER HEADING DETECTION PIPELINE")
            print("="*60)
            print(f"⚙️  Current mode: {self.config['pipeline_settings']['training_mode'].upper()}")
            print("0. ⚙️  Change configuration mode")
            print("1. 🌱 Train new model on labeled data")
            print("1a. 🎯 Train model with config selection")
            print("2. 📄 Process PDFs (generate predictions)")
            print("3. 🔄 Retrain with manual corrections")
            print("4. 📤 Generate JSON output")
            print("5. 🚀 Run full cycle (all steps)")
            print("6. 📊 Show status")
            print("7. 🔍 Run comprehensive accuracy test")
            print("8. 🧹 Clean working directories")
            print("9. 📋 Show current configuration")
            print("10. ❌ Exit")
            print()
            
            choice = input("Choose an option (0-10, 1a): ").strip()
            
            if choice == '0':
                self.change_configuration()
            
            elif choice == '1':
                df = self.load_labeled_data()
                if df is not None:
                    self.train_model(df)
                    self.save_model()
            
            elif choice == '1a':
                self.train_with_config_selection()
            
            elif choice == '2':
                max_pdfs = input("Max PDFs to process (Enter for all): ").strip()
                max_pdfs = int(max_pdfs) if max_pdfs.isdigit() else None
                self.process_pdf_batch(max_pdfs)
            
            elif choice == '3':
                self.retrain_with_corrections()
            
            elif choice == '4':
                process_input = input("Process 'input' folder? (y/N): ").strip().lower() == 'y'
                self.generate_json_output(process_input)
            
            elif choice == '5':
                max_pdfs = input("Max PDFs per cycle (Enter for all): ").strip()
                max_pdfs = int(max_pdfs) if max_pdfs.isdigit() else None
                self.run_full_cycle(max_pdfs)
            
            elif choice == '6':
                self.show_status()
            
            elif choice == '7':
                print("🔍 Running comprehensive accuracy test...")
                print("This will analyze your model's performance with detailed metrics.")
                print("It may take a few minutes to complete.")
                print()
                self.comprehensive_accuracy_test()
            
            elif choice == '8':
                self.clean_directories()
            
            elif choice == '9':
                self.show_configuration()
            
            elif choice == '10':
                print("👋 Goodbye!")
                break
            
            else:
                print("❌ Invalid choice. Please try again.")
    
    def change_configuration(self):
        """Change pipeline configuration"""
        print("\n🔧 CONFIGURATION MODES")
        print("-" * 40)
        print("1. 🚀 Fast Training (5-10 min)")
        print("   - Quick results, good for testing")
        print("   - 100 trees, basic features")
        print()
        print("2. ⚖️  Balanced Performance (15-20 min)")
        print("   - Good balance of speed and accuracy")
        print("   - 200 trees, standard features")
        print()
        print("3. 🎯 High Accuracy (30-45 min)")
        print("   - Best possible results")
        print("   - 500 trees, advanced features")
        print()
        print("4. � Advanced POS Features")
        print("   - High accuracy with POS features")
        print("   - Enhanced feature engineering")
        print()
        print("5. 🎯 High Accuracy v2")
        print("   - Extended high accuracy mode")
        print("   - 1000 trees, maximum features")
        print()
        print("6. �📄 Custom configuration file")
        print()
        
        choice = input("Choose configuration (1-6): ").strip()
        
        if choice == '1':
            config_path = os.path.join(self.base_dir, 'config_fast.json')
        elif choice == '2':
            config_path = os.path.join(self.base_dir, 'config.json')
        elif choice == '3':
            config_path = os.path.join(self.base_dir, 'config_high_accuracy.json')
        elif choice == '4':
            config_path = os.path.join(self.base_dir, 'config_advanced_pos.json')
        elif choice == '5':
            config_path = os.path.join(self.base_dir, 'config_high_accuracy_v2.json')
        elif choice == '6':
            config_path = input("Enter config file path: ").strip()
        else:
            print("❌ Invalid choice")
            return
        
        if os.path.exists(config_path):
            self.config = self.load_config(config_path)
            print(f"✅ Configuration changed to: {self.config['pipeline_settings']['training_mode']}")
        else:
            print(f"❌ Configuration file not found: {config_path}")
    
    def train_with_config_selection(self):
        """Train model with user-selected configuration"""
        print("\n🎯 TRAIN MODEL WITH CONFIG SELECTION")
        print("=" * 50)
        print("🔧 AVAILABLE CONFIGURATIONS:")
        print("-" * 40)
        print("1. 🚀 Fast Training (config_fast.json)")
        print("   - 100 trees, basic features, 5-10 min")
        print("   - Best for: Quick testing, small datasets")
        print()
        print("2. ⚖️  Balanced Performance (config.json)")
        print("   - 200 trees, standard features, 15-20 min")
        print("   - Best for: General use, medium datasets")
        print()
        print("3. 🎯 High Accuracy (config_high_accuracy.json)")
        print("   - 500 trees, advanced features, 30-45 min")
        print("   - Best for: Competition accuracy")
        print()
        print("4. 🚀 Advanced POS Features (config_advanced_pos.json)")
        print("   - 800 trees, POS features, 45-60 min")
        print("   - Best for: Mixed-language documents")
        print()
        print("5. 🎯 High Accuracy v2 (config_high_accuracy_v2.json)")
        print("   - 1000 trees, maximum features, 60+ min")
        print("   - Best for: Maximum possible accuracy")
        print()
        print("6. 📄 Custom configuration file")
        print("7. ❌ Cancel (keep current config)")
        print()
        
        choice = input("Choose configuration for training (1-7): ").strip()
        
        # Save current config to restore if needed
        original_config = self.config.copy()
        
        config_path = None
        if choice == '1':
            config_path = os.path.join(self.base_dir, 'config_fast.json')
        elif choice == '2':
            config_path = os.path.join(self.base_dir, 'config.json')
        elif choice == '3':
            config_path = os.path.join(self.base_dir, 'config_high_accuracy.json')
        elif choice == '4':
            config_path = os.path.join(self.base_dir, 'config_advanced_pos.json')
        elif choice == '5':
            config_path = os.path.join(self.base_dir, 'config_high_accuracy_v2.json')
        elif choice == '6':
            config_path = input("Enter custom config file path: ").strip()
            if not os.path.isabs(config_path):
                config_path = os.path.join(self.base_dir, config_path)
        elif choice == '7':
            print("❌ Training cancelled")
            return
        else:
            print("❌ Invalid choice")
            return
        
        if not os.path.exists(config_path):
            print(f"❌ Configuration file not found: {config_path}")
            return
        
        try:
            # Load the selected configuration
            print(f"\n📄 Loading configuration: {os.path.basename(config_path)}")
            self.config = self.load_config(config_path)
            print(f"✅ Configuration loaded: {self.config['pipeline_settings']['training_mode']}")
            
            # Reset model and vectorizer to ensure clean state
            self.model = None
            self.tfidf_vectorizer = None
            self.feature_columns = None
            
            # Load labeled data
            print("\n📚 Loading labeled data...")
            df = self.load_labeled_data()
            
            if df is None or len(df) == 0:
                print("❌ No labeled data found!")
                self.config = original_config  # Restore original config
                return
            
            print(f"✅ Loaded {len(df)} labeled samples")
            print(f"   Headings: {df['is_heading'].sum()}")
            print(f"   Percentage: {(df['is_heading'].sum() / len(df) * 100):.1f}%")
            
            # Show estimated training time
            trees = self.config['model_params']['n_estimators']
            features = self.config['tfidf_params']['max_features']
            if trees <= 100 and features <= 50:
                time_est = "5-10 minutes"
            elif trees <= 300 and features <= 100:
                time_est = "15-25 minutes"
            elif trees <= 600 and features <= 75:
                time_est = "30-45 minutes"
            else:
                time_est = "45-60+ minutes"
            
            print(f"\n⏱️  Estimated training time: {time_est}")
            print(f"🌳 Trees: {trees}")
            print(f"📝 TF-IDF features: {features}")
            
            # Confirm training
            confirm = input("\nProceed with training? (y/N): ").strip().lower()
            if confirm != 'y':
                print("❌ Training cancelled")
                self.config = original_config  # Restore original config
                return
            
            # Train the model with selected configuration
            print(f"\n🌳 Training model with {os.path.basename(config_path)}...")
            success = self.train_model(df)
            
            if success:
                # Save model with config-specific name
                config_name = os.path.splitext(os.path.basename(config_path))[0]
                version_name = f"{config_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                print(f"\n💾 Saving model as: {version_name}")
                self.save_model(version_name)
                self.save_model("latest")  # Also save as latest
                
                print("✅ Training completed successfully!")
                print(f"🎯 Model trained with: {self.config['pipeline_settings']['training_mode']}")
            else:
                print("❌ Training failed!")
                self.config = original_config  # Restore original config
                
        except Exception as e:
            print(f"❌ Training failed with error: {e}")
            import traceback
            traceback.print_exc()
            self.config = original_config  # Restore original config
    
    def show_configuration(self):
        """Show current configuration details"""
        print("\n📋 CURRENT CONFIGURATION")
        print("-" * 50)
        
        # Pipeline settings
        pipeline = self.config['pipeline_settings']
        print(f"🔧 Training Mode: {pipeline['training_mode']}")
        print(f"🎯 Confidence Threshold: {pipeline['confidence_threshold']}")
        print(f"📊 Min Heading %: {pipeline['min_heading_percentage']}")
        print()
        
        # Model parameters
        model = self.config['model_params']
        print(f"🌳 Trees: {model['n_estimators']}")
        print(f"📏 Max Depth: {model['max_depth']}")
        print(f"⚖️  Class Weight: {model['class_weight']}")
        print()
        
        # Feature settings
        tfidf = self.config['tfidf_params']
        print(f"📝 TF-IDF Features: {tfidf['max_features']}")
        print(f"🔤 N-gram Range: {tfidf['ngram_range']}")
        print()
        
        # SMOTE settings
        smote = self.config['smote_params']
        print(f"⚖️  SMOTE Sampling: {smote['sampling_strategy']}")
        print()
        
        # Estimated training time
        trees = model['n_estimators']
        features = tfidf['max_features']
        if trees <= 100 and features <= 50:
            time_est = "5-10 minutes"
        elif trees <= 300 and features <= 100:
            time_est = "15-25 minutes"
        else:
            time_est = "30-45 minutes"
        
        print(f"⏱️  Estimated Training Time: {time_est}")
    
    def show_status(self):
        """Show pipeline status"""
        print("\\n📊 PIPELINE STATUS")
        print("-" * 40)
        
        # Model status
        model_exists = os.path.exists(os.path.join(self.models_dir, "heading_model_latest.pkl"))
        print(f"🤖 Model trained: {'✅ Yes' if model_exists else '❌ No'}")
        
        # Data status
        csv_files = glob.glob(os.path.join(self.labelled_data_dir, "*.csv"))
        csv_files = [f for f in csv_files if not f.endswith('_with_predictions.csv')]
        print(f"📊 Labeled data files: {len(csv_files)}")
        
        pdf_files = list(Path(self.unprocessed_pdfs_dir).glob("*.pdf"))
        print(f"📄 Unprocessed PDFs: {len(pdf_files)}")
        
        input_pdfs = list(Path(self.input_dir).glob("*.pdf"))
        print(f"📥 Input PDFs: {len(input_pdfs)}")
        
        # Predictions status
        prediction_files = list(Path(self.predictions_dir).glob("*.csv"))
        print(f"🔮 Prediction files: {len(prediction_files)}")
        
        reviewed_files = list(Path(self.reviewed_dir).glob("*.csv"))
        print(f"👁️  Reviewed files: {len(reviewed_files)}")
        
        # Output status
        json_files = list(Path(self.output_dir).glob("*.json"))
        print(f"📤 JSON output files: {len(json_files)}")
        
        print(f"🔄 Current cycle: {self.current_cycle}")
    
    def clean_directories(self):
        """Clean working directories"""
        print("\\n🧹 CLEANING DIRECTORIES")
        
        confirm = input("Delete all predictions and reviewed files? (y/N): ").strip().lower()
        
        if confirm == 'y':
            import shutil
            
            # Clean predictions
            if os.path.exists(self.predictions_dir):
                shutil.rmtree(self.predictions_dir)
                os.makedirs(self.predictions_dir)
                print("✅ Cleaned predictions directory")
            
            # Clean reviewed
            if os.path.exists(self.reviewed_dir):
                shutil.rmtree(self.reviewed_dir)
                os.makedirs(self.reviewed_dir)
                print("✅ Cleaned reviewed directory")
            
            print("🧹 Cleanup completed!")
        else:
            print("❌ Cleanup cancelled")


def main():
    """Main function"""
    print("🤖 MASTER SEMI-AUTOMATED HEADING DETECTION PIPELINE")
    print("=" * 60)
    print("🎯 Features:")
    print("   ✅ Train on your labeled CSV data")
    print("   ✅ Process PDFs for predictions") 
    print("   ✅ Semi-automated review cycle")
    print("   ✅ Model retraining with corrections")
    print("   ✅ JSON output for competition")
    print("   ✅ Complete end-to-end workflow")
    print("   ✅ Single unified configuration")
    print()
    print("📖 Configuration: Uses config_main.json only")
    print("   See CONFIG_README.md for customization options")
    print()
    
    try:
        pipeline = MasterPipeline()
        pipeline.interactive_menu()
    except KeyboardInterrupt:
        print("\n👋 Pipeline interrupted by user")
    except Exception as e:
        logger.error(f"❌ Pipeline error: {e}")
        raise


if __name__ == "__main__":
    main()
