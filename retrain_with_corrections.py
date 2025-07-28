#!/usr/bin/env python3
"""
Retrain with Manual Corrections Script
=====================================

This script handles retraining the model using original labeled data 
plus manually corrected predictions.

Features:
✅ Load original labeled training data
✅ Load reviewed/corrected prediction files
✅ Combine datasets for enhanced training
✅ Retrain model with improved data
✅ Save updated model with versioning
✅ Track training cycles and improvements

Usage:
    python retrain_with_corrections.py

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

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelRetrainer:
    """Retrain model with manual corrections"""
    
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load configuration
        self.config = self.load_config()
        
        # Core directories
        self.labelled_data_dir = os.path.join(self.base_dir, self.config['directories']['labelled_data'])
        self.reviewed_dir = os.path.join(self.base_dir, self.config['directories']['reviewed'])
        self.models_dir = os.path.join(self.base_dir, self.config['directories']['models'])
        
        # Create directories
        os.makedirs(self.reviewed_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Model components
        self.model = None
        self.tfidf_vectorizer = None
        self.feature_columns = None
        self.optimal_threshold = self.config['pipeline_settings']['confidence_threshold']
        self.current_cycle = 0
        
        logger.info("🔄 Model Retrainer initialized!")
        logger.info(f"📁 Labeled data: {self.labelled_data_dir}")
        logger.info(f"📁 Reviewed data: {self.reviewed_dir}")
        logger.info(f"📁 Models: {self.models_dir}")
    
    def load_config(self):
        """Load configuration from the main config file"""
        config_path = os.path.join(self.base_dir, 'config_main.json')
        
        if not os.path.exists(config_path):
            logger.error(f"❌ Main config file not found: {config_path}")
            raise FileNotFoundError(f"Main configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"✅ Loaded configuration: {config_path}")
            return config
        except Exception as e:
            logger.error(f"❌ Error loading config: {e}")
            raise
    
    def load_labeled_data(self, min_heading_percentage=None):
        """Load and combine all labeled CSV data"""
        if min_heading_percentage is None:
            min_heading_percentage = self.config['pipeline_settings']['min_heading_percentage']
            
        logger.info("📊 Loading original labeled training data...")
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
                df['source_type'] = 'original'
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
        
        logger.info(f"📊 Original dataset: {len(combined_df)} total blocks")
        logger.info(f"✅ Used {total_files} files, skipped {skipped_files} files")
        logger.info(f"🎯 Total headings: {combined_df['is_heading'].sum()} ({(combined_df['is_heading'].sum()/len(combined_df)*100):.1f}%)")
        
        return combined_df
    
    def load_reviewed_data(self):
        """Load reviewed corrections from the reviewed directory"""
        logger.info("👁️  Loading reviewed corrections...")
        
        reviewed_files = list(Path(self.reviewed_dir).glob("*_predictions.csv"))
        
        if not reviewed_files:
            logger.warning("⚠️  No reviewed files found for retraining")
            return None
        
        logger.info(f"📄 Found {len(reviewed_files)} reviewed files")
        
        reviewed_dataframes = []
        total_corrections = 0
        
        for file_path in reviewed_files:
            try:
                df = pd.read_csv(file_path)
                
                if 'is_heading' not in df.columns:
                    logger.warning(f"⚠️  Skipping {os.path.basename(file_path)} - no 'is_heading' column")
                    continue
                
                # Add metadata
                df['source_file'] = f"reviewed_{os.path.basename(file_path)}"
                df['source_type'] = 'reviewed'
                
                reviewed_dataframes.append(df)
                
                heading_count = df['is_heading'].sum()
                total_corrections += len(df)
                
                logger.info(f"✅ {os.path.basename(file_path)}: {len(df)} blocks, {heading_count} headings ({heading_count/len(df)*100:.1f}%)")
                
            except Exception as e:
                logger.error(f"❌ Error loading {file_path}: {e}")
        
        if not reviewed_dataframes:
            logger.error("❌ No valid reviewed data found")
            return None
        
        combined_reviewed_df = pd.concat(reviewed_dataframes, ignore_index=True)
        
        logger.info(f"📊 Reviewed dataset: {len(combined_reviewed_df)} total blocks")
        logger.info(f"🎯 Total reviewed headings: {combined_reviewed_df['is_heading'].sum()} ({(combined_reviewed_df['is_heading'].sum()/len(combined_reviewed_df)*100):.1f}%)")
        
        return combined_reviewed_df
    
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
    
    def validate_and_enhance_dataframe(self, df):
        """Validate and enhance dataframe with missing features before training"""
        logger.info("🔍 Validating and enhancing dataframe features...")
        
        # Check for distance_to_previous_heading
        if 'distance_to_previous_heading' not in df.columns or df['distance_to_previous_heading'].isna().all():
            logger.info("📏 Calculating missing distance_to_previous_heading...")
            df = self._calculate_distance_to_previous_heading(df)
        elif df['distance_to_previous_heading'].isna().any():
            logger.info("📏 Recalculating partially missing distance_to_previous_heading...")
            df = self._calculate_distance_to_previous_heading(df)
        
        # Check for POS features
        pos_features = ['num_nouns', 'num_verbs', 'num_adjs', 'num_advs', 'num_propn', 'num_pronouns', 'num_other_pos']
        missing_pos_features = [f for f in pos_features if f not in df.columns or df[f].isna().all()]
        
        if missing_pos_features:
            logger.info(f"🏷️  Missing POS features: {missing_pos_features}")
            df = self._add_missing_pos_features(df)
        
        # Ensure required columns exist with defaults
        required_columns = {
            'font_size': 12.0,
            'page': 1,
            'x0': 0.0, 'y0': 0.0, 'x1': 100.0, 'y1': 20.0,
            'word_count': df['text'].str.split().str.len().fillna(1),
            'line_position_on_page': 0.5,
            'relative_font_size': 1.0,
            'line_spacing_above': 0.0
        }
        
        for col, default_value in required_columns.items():
            if col not in df.columns:
                if hasattr(default_value, '__iter__') and not isinstance(default_value, str):
                    df[col] = default_value
                else:
                    df[col] = default_value
                logger.info(f"➕ Added missing column: {col}")
        
        logger.info(f"✅ Dataframe validated with {len(df.columns)} columns")
        return df
    
    def _calculate_distance_to_previous_heading(self, df, heading_col='is_heading', distance_col='distance_to_previous_heading'):
        """Calculate distance to previous heading for each row"""
        last_heading_idx = None
        distances = []
        
        for idx, is_heading in enumerate(df[heading_col]):
            if is_heading == 1 or is_heading == '1' or is_heading == True:
                distances.append(0)
                last_heading_idx = idx
            elif last_heading_idx is not None:
                distances.append(idx - last_heading_idx)
            else:
                distances.append(None)
        
        df[distance_col] = distances
        return df
    
    def _add_missing_pos_features(self, df):
        """Add missing POS features to dataframe"""
        pos_features = ['num_nouns', 'num_verbs', 'num_adjs', 'num_advs', 'num_propn', 'num_pronouns', 'num_other_pos']
        
        # Initialize missing features with zeros
        for feature in pos_features:
            if feature not in df.columns:
                df[feature] = 0
        
        # Try to calculate POS features if spaCy is available
        try:
            import spacy
            from langdetect import detect
            
            # Try to load the spaCy model
            try:
                nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("⚠️  English spaCy model not found, using zero-filled POS features")
                return df
            
            logger.info("🏷️  Computing missing POS features...")
            
            # Process text for POS features (sample first 100 rows to avoid long processing)
            sample_size = min(100, len(df))
            for idx in range(sample_size):
                text = df.iloc[idx]['text']
                if pd.isna(text) or len(str(text).strip()) == 0:
                    continue
                    
                try:
                    text_str = str(text)
                    
                    # Quick language detection for longer texts
                    if len(text_str.strip()) > 10:
                        try:
                            lang = detect(text_str)
                        except:
                            lang = 'en'
                    else:
                        lang = 'en'
                    
                    # Only process English text with spaCy
                    if lang == 'en' and len(text_str.strip()) > 0:
                        doc = nlp(text_str)
                        
                        # Count POS tags
                        df.at[idx, 'num_nouns'] = sum(1 for token in doc if token.pos_ == 'NOUN')
                        df.at[idx, 'num_verbs'] = sum(1 for token in doc if token.pos_ == 'VERB')
                        df.at[idx, 'num_adjs'] = sum(1 for token in doc if token.pos_ == 'ADJ')
                        df.at[idx, 'num_advs'] = sum(1 for token in doc if token.pos_ == 'ADV')
                        df.at[idx, 'num_propn'] = sum(1 for token in doc if token.pos_ == 'PROPN')
                        df.at[idx, 'num_pronouns'] = sum(1 for token in doc if token.pos_ == 'PRON')
                        
                        # Calculate other POS tags
                        total_pos = len([t for t in doc if t.pos_ != 'SPACE'])
                        counted_pos = (df.at[idx, 'num_nouns'] + df.at[idx, 'num_verbs'] + 
                                     df.at[idx, 'num_adjs'] + df.at[idx, 'num_advs'] + 
                                     df.at[idx, 'num_propn'] + df.at[idx, 'num_pronouns'])
                        df.at[idx, 'num_other_pos'] = max(0, total_pos - counted_pos)
                        
                except Exception:
                    # Skip problematic text, keep zeros
                    continue
            
            logger.info(f"✅ POS features computed for sample of {sample_size} rows")
            
        except ImportError:
            logger.warning("⚠️  spaCy/langdetect not available, using zero-filled POS features")
        
        return df
    
    def prepare_features(self, df):
        """Prepare enhanced features for training"""
        logger.info("🔧 Preparing enhanced features for retraining...")
        
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
                if hasattr(center_mask, 'astype'):
                    features['center_aligned'] = center_mask.astype(int)
                else:
                    features['center_aligned'] = int(center_mask)
            except Exception as e:
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
                        features[f'font_ge_{percentile}p'] = int(comparison_result)
                
                # Font size statistics
                features['font_size_normalized'] = (features['font_size'] - features['font_size'].min()) / (features['font_size'].max() - features['font_size'].min() + 1)
                features['font_size_zscore'] = (features['font_size'] - features['font_size'].mean()) / (features['font_size'].std() + 1)
                
                largest_font_mask = (features['font_size'] == features['font_size'].max())
                if hasattr(largest_font_mask, 'astype'):
                    features['is_largest_font'] = largest_font_mask.astype(int)
                else:
                    features['is_largest_font'] = int(largest_font_mask)
                
                above_median_mask = (features['font_size'] > features['font_size'].median())
                if hasattr(above_median_mask, 'astype'):
                    features['above_median_font'] = above_median_mask.astype(int)
                else:
                    features['above_median_font'] = int(above_median_mask)
                    
            except Exception as e:
                logger.warning(f"⚠️  Error in font feature calculation: {e}")
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
            features['is_first_page'] = (features['page_num'] == 1).astype(int)
            features['is_early_page'] = (features['page_num'] <= 3).astype(int)
        elif 'page' in df.columns:
            features['page_num'] = df['page'].fillna(1)
            features['is_first_page'] = (features['page_num'] == 1).astype(int)
            features['is_early_page'] = (features['page_num'] <= 3).astype(int)
        else:
            features['page_num'] = 1
            features['is_first_page'] = 1
            features['is_early_page'] = 1
        
        # Add original CSV features if available
        csv_features = ['distance_to_previous_heading', 'line_spacing_above', 'relative_font_size', 
                       'line_position_on_page', 'bold', 'italic', 'underline']
        
        for feature in csv_features:
            if feature in df.columns:
                # Use actual CSV values
                features[feature] = df[feature].fillna(0)
                logger.debug(f"✅ Using CSV feature: {feature}")
            else:
                # Provide sensible defaults
                if feature == 'distance_to_previous_heading':
                    features[feature] = 0  # Will be calculated in validation
                elif feature == 'line_spacing_above':
                    features[feature] = 0
                elif feature == 'relative_font_size':
                    features[feature] = 1.0
                elif feature == 'line_position_on_page':
                    features[feature] = 0.5
                elif feature in ['bold', 'italic', 'underline']:
                    features[feature] = 0
                logger.debug(f"➕ Added default for missing feature: {feature}")
        
        # Enhanced TF-IDF features
        logger.info("📝 Computing enhanced TF-IDF features...")
        tfidf_config = self.config['tfidf_params']
        
        # Create new TF-IDF vectorizer for training
        logger.info("🔧 Creating new TF-IDF vectorizer for retraining...")
        
        # Adaptive configuration for small datasets
        dataset_size = len(df)
        if dataset_size < 100:
            min_df = 1
            max_df = 1.0
            max_features = min(20, dataset_size)
            logger.info("🔧 Using adaptive TF-IDF for small dataset")
        elif dataset_size < 1000:
            min_df = 1
            max_df = 0.95
            max_features = min(tfidf_config['max_features'], dataset_size // 2)
        else:
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
        
        # Add TF-IDF features
        tfidf_feature_names = [f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])]
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_feature_names)
        
        # Convert features dict to DataFrame for POS processing
        features_df = pd.DataFrame(features)
        
        # Add POS-based features if enabled
        pos_features_enabled = self.config.get('feature_engineering', {}).get('pos_features', False)
        pos_features = ['num_nouns', 'num_verbs', 'num_adjs', 'num_advs', 'num_propn', 'num_pronouns', 'num_other_pos']
        
        # Check if POS features already exist in CSV
        csv_has_pos = all(col in df.columns for col in pos_features)
        
        if csv_has_pos:
            logger.info("📄 Using POS features from CSV data...")
            for feature in pos_features:
                features_df[feature] = df[feature].fillna(0)
        elif pos_features_enabled:
            logger.info("🏷️  Computing POS features with spaCy...")
            features_df = self._add_pos_features(df, features_df)
        else:
            # Add zero-filled POS features for consistency
            logger.info("➕ Adding zero-filled POS features...")
            for feature in pos_features:
                features_df[feature] = 0
        
        # Add derived POS features
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
        logger.info("🌳 Training enhanced heading detection model with corrections...")
        
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
            
            if len(X_train) < 100:
                adjusted_strategy = min(original_strategy, 0.8)
                logger.info(f"   📉 Adjusting SMOTE strategy from {original_strategy} to {adjusted_strategy} for small dataset")
            else:
                adjusted_strategy = original_strategy
            
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
        
        # Get model configuration
        model_config = self.config['model_params']
        
        # Train Random Forest with enhanced configuration
        logger.info(f"🔧 Enhanced Model Configuration:")
        logger.info(f"   🌳 Trees (n_estimators): {model_config['n_estimators']}")
        logger.info(f"   📏 Max Depth: {model_config['max_depth']}")
        logger.info(f"   🍃 Min Samples Split: {model_config['min_samples_split']}")
        logger.info(f"   🍂 Min Samples Leaf: {model_config['min_samples_leaf']}")
        logger.info(f"   ⚖️  Class Weight: {model_config['class_weight']}")
        
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
            n_jobs=-1,
            class_weight={int(k): v for k, v in model_config['class_weight'].items()},
            verbose=1,
            warm_start=False
        )
        
        logger.info(f"🏋️  Retraining Random Forest with {model_config['n_estimators']} trees...")
        
        # Train with progress tracking
        self.model.fit(X_train_balanced, y_train_balanced)
        
        training_time = time.time() - start_time
        logger.info(f"⏱️  Retraining completed in {training_time:.1f} seconds ({training_time/60:.1f} minutes)")
        
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
            logger.info(f"📈 ROC AUC Score: {test_auc:.3f}")
        except Exception as e:
            logger.warning(f"⚠️  ROC AUC calculation failed: {e}")
            test_auc = 0.5
        
        logger.info(f"✅ Enhanced model retraining completed!")
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
    
    def _find_enhanced_threshold(self, y_true, y_proba):
        """Find optimal threshold with granular search focused on recall"""
        thresholds = np.arange(0.1, 0.8, 0.02)
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
        """Save the retrained model"""
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
            
            logger.info(f"💾 Retrained model saved: {model_path}")
            logger.info(f"💾 TF-IDF vectorizer saved: {vectorizer_path}")
            logger.info(f"🔧 Saved {len(self.feature_columns)} feature columns")
            return model_path
        except Exception as e:
            logger.error(f"❌ Error saving model: {e}")
            return None
    
    def retrain_with_corrections(self):
        """Retrain model using original data + corrected predictions"""
        logger.info("\n" + "="*70)
        logger.info("🔄 RETRAINING WITH CORRECTIONS")
        logger.info("="*70)
        
        # Load original labeled data
        original_df = self.load_labeled_data(min_heading_percentage=0.5)
        
        if original_df is None:
            logger.error("❌ Failed to load original labeled data")
            return False
        
        # Load reviewed corrections
        reviewed_df = self.load_reviewed_data()
        
        if reviewed_df is None:
            logger.warning("⚠️  No reviewed files found for retraining")
            logger.info("🔄 Training with original data only...")
            combined_df = original_df
        else:
            # Combine original + reviewed data
            all_dataframes = [original_df, reviewed_df]
            combined_df = pd.concat(all_dataframes, ignore_index=True)
        
        logger.info(f"📊 Combined training data: {len(combined_df)} blocks")
        logger.info(f"🎯 Total headings: {combined_df['is_heading'].sum()} ({combined_df['is_heading'].mean()*100:.1f}%)")
        
        # Validate and enhance dataframe with missing features
        combined_df = self.validate_and_enhance_dataframe(combined_df)
        
        # Show data source breakdown
        if 'source_type' in combined_df.columns:
            source_breakdown = combined_df['source_type'].value_counts()
            logger.info(f"📊 Data sources:")
            for source, count in source_breakdown.items():
                headings = combined_df[combined_df['source_type'] == source]['is_heading'].sum()
                logger.info(f"   {source}: {count} blocks, {headings} headings ({headings/count*100:.1f}%)")
        
        # Retrain model
        self.current_cycle += 1
        results = self.train_model(combined_df)
        
        if results:
            # Save updated model
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            cycle_version = f"retrained_cycle_{self.current_cycle}_{timestamp}"
            model_path = self.save_model(cycle_version)
            self.save_model("latest")  # Also save as latest
            
            logger.info(f"✅ Retraining completed! (Cycle {self.current_cycle})")
            logger.info(f"💾 Model saved as: {cycle_version}")
            return True
        else:
            logger.error("❌ Retraining failed!")
            return False
    
    def show_status(self):
        """Show retrainer status"""
        print("\n📊 RETRAINER STATUS")
        print("-" * 40)
        
        # Data status
        csv_files = glob.glob(os.path.join(self.labelled_data_dir, "*.csv"))
        csv_files = [f for f in csv_files if not f.endswith('_with_predictions.csv')]
        print(f"📊 Original labeled data files: {len(csv_files)}")
        
        reviewed_files = list(Path(self.reviewed_dir).glob("*_predictions.csv"))
        print(f"👁️  Reviewed correction files: {len(reviewed_files)}")
        
        print(f"🔄 Current cycle: {self.current_cycle}")
    
    def interactive_menu(self):
        """Interactive menu for retraining operations"""
        while True:
            print("\n" + "="*60)
            print("🔄 MODEL RETRAINING SCRIPT")
            print("="*60)
            print("1. 🔄 Retrain with corrections")
            print("2. 📊 Show original data statistics")
            print("3. 👁️  Show reviewed data statistics")
            print("4. 📊 Show status")
            print("5. 🧹 Clean reviewed directory")
            print("6. ❌ Exit")
            print()
            
            choice = input("Choose an option (1-6): ").strip()
            
            if choice == '1':
                success = self.retrain_with_corrections()
                if success:
                    print("✅ Retraining completed successfully!")
                else:
                    print("❌ Retraining failed!")
            
            elif choice == '2':
                df = self.load_labeled_data()
                if df is not None:
                    print(f"\n📊 Original Data Statistics:")
                    print(f"   Total samples: {len(df):,}")
                    print(f"   Total headings: {df['is_heading'].sum():,}")
                    print(f"   Heading percentage: {(df['is_heading'].sum()/len(df)*100):.2f}%")
                    print(f"   Unique files: {df['source_file'].nunique()}")
            
            elif choice == '3':
                df = self.load_reviewed_data()
                if df is not None:
                    print(f"\n👁️  Reviewed Data Statistics:")
                    print(f"   Total samples: {len(df):,}")
                    print(f"   Total headings: {df['is_heading'].sum():,}")
                    print(f"   Heading percentage: {(df['is_heading'].sum()/len(df)*100):.2f}%")
                    print(f"   Unique files: {df['source_file'].nunique()}")
                else:
                    print("⚠️  No reviewed data found")
            
            elif choice == '4':
                self.show_status()
            
            elif choice == '5':
                confirm = input("Delete all reviewed files? (y/N): ").strip().lower()
                if confirm == 'y':
                    import shutil
                    if os.path.exists(self.reviewed_dir):
                        shutil.rmtree(self.reviewed_dir)
                        os.makedirs(self.reviewed_dir)
                        print("✅ Cleaned reviewed directory")
                else:
                    print("❌ Cleanup cancelled")
            
            elif choice == '6':
                print("👋 Goodbye!")
                break
            
            else:
                print("❌ Invalid choice. Please try again.")


def main():
    """Main function"""
    print("🔄 MODEL RETRAINING WITH CORRECTIONS SCRIPT")
    print("=" * 50)
    print("🎯 Features:")
    print("   ✅ Load original labeled training data")
    print("   ✅ Load reviewed/corrected prediction files")
    print("   ✅ Combine datasets for enhanced training")
    print("   ✅ Retrain model with improved data")
    print("   ✅ Save updated model with versioning")
    print("   ✅ Track training cycles and improvements")
    print()
    
    try:
        retrainer = ModelRetrainer()
        retrainer.interactive_menu()
    except KeyboardInterrupt:
        print("\n👋 Retraining interrupted by user")
    except Exception as e:
        logger.error(f"❌ Retraining error: {e}")
        raise


if __name__ == "__main__":
    main()
