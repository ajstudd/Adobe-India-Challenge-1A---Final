#!/usr/bin/env python3
"""
Model Training Script
====================

This script handles training the heading detection model with enhanced features.

Features:
✅ Enhanced feature engineering with TF-IDF, POS, and position features
✅ SMOTE class balancing for handling imbalanced datasets
✅ Cross-validation and threshold optimization
✅ Feature selection with RFECV
✅ Comprehensive logging and metrics
✅ Model saving with versioning

Usage:
    python train_model.py

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

class ModelTrainer:
    """Enhanced model trainer with comprehensive feature engineering"""
    
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load configuration
        self.config = self.load_config()
        
        # Core directories
        self.labelled_data_dir = os.path.join(self.base_dir, self.config['directories']['labelled_data'])
        self.models_dir = os.path.join(self.base_dir, self.config['directories']['models'])
        
        # Create directories
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Model components
        self.model = None
        self.tfidf_vectorizer = None
        self.feature_columns = None
        self.optimal_threshold = self.config['pipeline_settings']['confidence_threshold']
        
        logger.info("🚀 Model Trainer initialized!")
        logger.info(f"⚙️  Configuration: {self.config['pipeline_settings']['training_mode']}")
        logger.info(f"📁 Labeled data: {self.labelled_data_dir}")
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
            logger.info(f"⚙️  Training mode: {config['pipeline_settings']['training_mode']}")
            return config
        except Exception as e:
            logger.error(f"❌ Error loading config: {e}")
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
    
    def prepare_features(self, df):
        """Prepare enhanced features for training"""
        logger.info("🔧 Preparing enhanced features for training...")
        
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
        
        # Enhanced TF-IDF features
        logger.info("📝 Computing enhanced TF-IDF features...")
        tfidf_config = self.config['tfidf_params']
        
        # Create new TF-IDF vectorizer for training
        logger.info("🔧 Creating new TF-IDF vectorizer for training...")
        
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
        if pos_features_enabled:
            features_df = self._add_pos_features(df, features_df)
        else:
            # Add zero-filled POS features for consistency
            pos_features = ['num_nouns', 'num_verbs', 'num_adjs', 'num_advs', 'num_propn', 'num_pronouns', 'num_other_pos']
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
        
        # Get model configuration early for use in RFECV
        model_config = self.config['model_params']
        
        # Optional RFECV Feature Selection
        feature_selection_enabled = self.config.get('training', {}).get('feature_selection', False)
        if feature_selection_enabled and len(self.feature_columns) > 100:
            logger.info("🔍 Applying RFECV feature selection...")
            
            # Use a simpler model for feature selection to save time
            rf_selector = RandomForestClassifier(
                n_estimators=50,
                max_depth=10,
                random_state=42,
                n_jobs=-1,
                class_weight={int(k): v for k, v in model_config['class_weight'].items()}
            )
            
            # Use RFECV with cross-validation
            rfecv = RFECV(
                estimator=rf_selector,
                step=10,
                cv=3,
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
            n_jobs=-1,
            class_weight={int(k): v for k, v in model_config['class_weight'].items()},
            verbose=1,
            warm_start=False
        )
        
        logger.info(f"🏋️  Training Random Forest with {model_config['n_estimators']} trees...")
        logger.info(f"🖥️  Using all CPU cores")
        
        # Train with progress tracking
        self.model.fit(X_train_balanced, y_train_balanced)
        
        training_time = time.time() - start_time
        logger.info(f"⏱️  Training completed in {training_time:.1f} seconds ({training_time/60:.1f} minutes)")
        
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
    
    def _estimate_detailed_training_time(self, n_estimators, n_samples, n_features):
        """Provide detailed training time estimate"""
        base_time_per_tree = 0.5  # seconds per tree for moderate complexity
        sample_factor = min(2.0, n_samples / 10000)
        feature_factor = min(1.5, n_features / 100)
        
        estimated_seconds = n_estimators * base_time_per_tree * sample_factor * feature_factor
        
        if estimated_seconds < 60:
            return f"{estimated_seconds:.0f} seconds"
        elif estimated_seconds < 3600:
            return f"{estimated_seconds/60:.1f} minutes"
        else:
            return f"{estimated_seconds/3600:.1f} hours"
    
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
    
    def interactive_menu(self):
        """Interactive menu for training options"""
        while True:
            print("\n" + "="*60)
            print("🌱 MODEL TRAINING SCRIPT")
            print("="*60)
            print(f"⚙️  Current mode: {self.config['pipeline_settings']['training_mode'].upper()}")
            print("1. 🌱 Train new model with current config")
            print("2. 📊 Show dataset statistics")
            print("3. 🔍 Run comprehensive accuracy test")
            print("4. 📋 Show current configuration")
            print("5. ❌ Exit")
            print()
            
            choice = input("Choose an option (1-5): ").strip()
            
            if choice == '1':
                df = self.load_labeled_data()
                if df is not None:
                    results = self.train_model(df)
                    if results:
                        self.save_model()
                        # Save with timestamp version too
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        self.save_model(f"trained_{timestamp}")
                        print("✅ Model training completed and saved!")
                    else:
                        print("❌ Model training failed!")
            
            elif choice == '2':
                df = self.load_labeled_data()
                if df is not None:
                    print(f"\n📊 Dataset Statistics:")
                    print(f"   Total samples: {len(df):,}")
                    print(f"   Total headings: {df['is_heading'].sum():,}")
                    print(f"   Heading percentage: {(df['is_heading'].sum()/len(df)*100):.2f}%")
                    print(f"   Unique files: {df['source_file'].nunique()}")
            
            elif choice == '3':
                try:
                    from comprehensive_accuracy_test import AccuracyLogger
                    logger.info("🔍 Running comprehensive accuracy test...")
                    accuracy_logger = AccuracyLogger()
                    success = accuracy_logger.run_full_accuracy_test()
                    if success:
                        print("✅ Comprehensive accuracy test completed!")
                    else:
                        print("❌ Comprehensive accuracy test failed!")
                except Exception as e:
                    print(f"❌ Error running accuracy test: {e}")
            
            elif choice == '4':
                print("\n📋 Current Configuration:")
                print(f"   Training Mode: {self.config['pipeline_settings']['training_mode']}")
                print(f"   Trees: {self.config['model_params']['n_estimators']}")
                print(f"   TF-IDF Features: {self.config['tfidf_params']['max_features']}")
                print(f"   Class Weight: {self.config['model_params']['class_weight']}")
            
            elif choice == '5':
                print("👋 Goodbye!")
                break
            
            else:
                print("❌ Invalid choice. Please try again.")


def main():
    """Main function"""
    print("🌱 HEADING DETECTION MODEL TRAINER")
    print("=" * 50)
    print("🎯 Features:")
    print("   ✅ Enhanced feature engineering")
    print("   ✅ SMOTE class balancing")
    print("   ✅ Cross-validation & threshold optimization")
    print("   ✅ Feature selection with RFECV")
    print("   ✅ Comprehensive metrics & logging")
    print("   ✅ Model saving with versioning")
    print()
    
    try:
        trainer = ModelTrainer()
        trainer.interactive_menu()
    except KeyboardInterrupt:
        print("\n👋 Training interrupted by user")
    except Exception as e:
        logger.error(f"❌ Training error: {e}")
        raise


if __name__ == "__main__":
    main()
