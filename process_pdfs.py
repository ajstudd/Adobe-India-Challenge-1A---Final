#!/usr/bin/env python3
"""
PDF Processing and Prediction Script
===================================

This script handles processing PDFs and generating heading predictions.

Features:
✅ Extract blocks from PDFs using PyMuPDF or existing extraction scripts
✅ Generate heading predictions using trained model
✅ Feature alignment and validation
✅ Batch processing of multiple PDFs
✅ Save predictions for manual review
✅ Comprehensive error handling and logging

Usage:
    python process_pdfs.py

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
from sklearn.feature_extraction.text import TfidfVectorizer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFProcessor:
    """PDF processor for generating heading predictions"""
    
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load configuration
        self.config = self.load_config()
        
        # Core directories
        self.unprocessed_pdfs_dir = os.path.join(self.base_dir, self.config['directories']['unprocessed_pdfs'])
        self.input_dir = os.path.join(self.base_dir, self.config['directories']['input'])
        self.predictions_dir = os.path.join(self.base_dir, self.config['directories']['predictions'])
        self.models_dir = os.path.join(self.base_dir, self.config['directories']['models'])
        
        # Create directories
        for dir_path in [self.predictions_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Model components
        self.model = None
        self.tfidf_vectorizer = None
        self.feature_columns = None
        self.optimal_threshold = self.config['pipeline_settings']['confidence_threshold']
        
        logger.info("📄 PDF Processor initialized!")
        logger.info(f"📁 Unprocessed PDFs: {self.unprocessed_pdfs_dir}")
        logger.info(f"📁 Input PDFs: {self.input_dir}")
        logger.info(f"📁 Predictions: {self.predictions_dir}")
    
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
                                    'x0': span['bbox'][0],
                                    'y0': span['bbox'][1],
                                    'x1': span['bbox'][2],
                                    'y1': span['bbox'][3],
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
        """Prepare enhanced features for prediction (using existing vectorizer)"""
        logger.info("🔧 Preparing enhanced features for prediction...")
        
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
        
        # Enhanced TF-IDF features (using existing vectorizer)
        logger.info("📝 Computing TF-IDF features using existing vectorizer...")
        
        if self.tfidf_vectorizer is None:
            logger.error("❌ TF-IDF vectorizer is None! This is a critical feature mismatch.")
            logger.error("   The model was trained with a different TF-IDF configuration.")
            logger.error("   Please retrain the model or load the correct vectorizer.")
            raise ValueError("Feature mismatch: TF-IDF vectorizer missing for loaded model")
        
        # Use existing vectorizer for prediction (transform only, don't fit)
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
        
        pos_feature_count = len([c for c in feature_df.columns if any(pos in c for pos in ['num_', 'ratio', 'pos_', 'noun', 'verb', 'adj'])])
        logger.info(f"✅ Added {pos_feature_count} total POS and derived features")
        
        logger.info(f"✅ Prepared {len(feature_df.columns)} enhanced features")
        logger.info(f"   📝 Text features: {len([c for c in feature_df.columns if 'text_' in c or 'word_' in c or 'char_' in c])}")
        logger.info(f"   📏 Font features: {len([c for c in feature_df.columns if 'font_' in c])}")
        logger.info(f"   📍 Position features: {len([c for c in feature_df.columns if 'position' in c or 'aligned' in c])}")
        logger.info(f"   🏷️  POS features: {len([c for c in feature_df.columns if 'num_' in c or 'pos' in c or 'ratio' in c])}")
        logger.info(f"   📄 TF-IDF features: {len(tfidf_feature_names)}")
        
        return feature_df
    
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
    
    def process_pdf_batch(self, source_dir=None, max_pdfs=None):
        """Process all PDFs in specified directory"""
        logger.info("\n" + "="*70)
        logger.info("📄 PROCESSING PDF BATCH FOR PREDICTIONS")
        logger.info("="*70)
        
        if self.model is None:
            logger.error("❌ No model loaded. Load a model first.")
            return False
        
        # Determine source directory
        if source_dir == "input":
            pdf_dir = self.input_dir
        else:
            pdf_dir = self.unprocessed_pdfs_dir
        
        pdf_files = list(Path(pdf_dir).glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"⚠️  No PDF files found in {pdf_dir}")
            return True
        
        if max_pdfs:
            pdf_files = pdf_files[:max_pdfs]
        
        logger.info(f"📄 Found {len(pdf_files)} PDF files to process in {pdf_dir}")
        
        successful = 0
        failed = 0
        
        for i, pdf_path in enumerate(pdf_files):
            pdf_name = pdf_path.stem
            logger.info(f"\n--- Processing PDF {i+1}/{len(pdf_files)}: {pdf_name} ---")
            
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
        
        logger.info(f"\n🎯 PDF processing complete!")
        logger.info(f"✅ Successfully processed: {successful} PDFs")
        logger.info(f"❌ Failed: {failed} PDFs")
        logger.info(f"📁 Predictions saved to: {self.predictions_dir}")
        
        return successful > 0
    
    def show_status(self):
        """Show processor status"""
        print("\n📊 PDF PROCESSOR STATUS")
        print("-" * 40)
        
        # Model status
        model_loaded = self.model is not None
        print(f"🤖 Model loaded: {'✅ Yes' if model_loaded else '❌ No'}")
        
        if model_loaded:
            print(f"🔧 Optimal threshold: {self.optimal_threshold}")
            print(f"📊 Expected features: {len(self.feature_columns) if self.feature_columns else 'Unknown'}")
        
        # PDF files status
        unprocessed_pdfs = list(Path(self.unprocessed_pdfs_dir).glob("*.pdf"))
        print(f"📄 Unprocessed PDFs: {len(unprocessed_pdfs)}")
        
        input_pdfs = list(Path(self.input_dir).glob("*.pdf"))
        print(f"📥 Input PDFs: {len(input_pdfs)}")
        
        # Predictions status
        prediction_files = list(Path(self.predictions_dir).glob("*.csv"))
        print(f"🔮 Prediction files: {len(prediction_files)}")
    
    def interactive_menu(self):
        """Interactive menu for PDF processing operations"""
        while True:
            print("\n" + "="*60)
            print("📄 PDF PROCESSING SCRIPT")
            print("="*60)
            print("1. 📤 Load model")
            print("2. 📄 Process unprocessed PDFs")
            print("3. 📥 Process input PDFs")
            print("4. 📊 Show status")
            print("5. 🧹 Clean predictions directory")
            print("6. ❌ Exit")
            print()
            
            choice = input("Choose an option (1-6): ").strip()
            
            if choice == '1':
                version = input("Model version (latest/specific): ").strip()
                if not version:
                    version = "latest"
                
                if self.load_model(version):
                    print("✅ Model loaded successfully!")
                else:
                    print("❌ Failed to load model!")
            
            elif choice == '2':
                if self.model is None:
                    print("❌ No model loaded. Please load a model first.")
                    continue
                
                max_pdfs = input("Max PDFs to process (Enter for all): ").strip()
                max_pdfs = int(max_pdfs) if max_pdfs.isdigit() else None
                self.process_pdf_batch("unprocessed", max_pdfs)
            
            elif choice == '3':
                if self.model is None:
                    print("❌ No model loaded. Please load a model first.")
                    continue
                
                max_pdfs = input("Max PDFs to process (Enter for all): ").strip()
                max_pdfs = int(max_pdfs) if max_pdfs.isdigit() else None
                self.process_pdf_batch("input", max_pdfs)
            
            elif choice == '4':
                self.show_status()
            
            elif choice == '5':
                confirm = input("Delete all prediction files? (y/N): ").strip().lower()
                if confirm == 'y':
                    import shutil
                    if os.path.exists(self.predictions_dir):
                        shutil.rmtree(self.predictions_dir)
                        os.makedirs(self.predictions_dir)
                        print("✅ Cleaned predictions directory")
                else:
                    print("❌ Cleanup cancelled")
            
            elif choice == '6':
                print("👋 Goodbye!")
                break
            
            else:
                print("❌ Invalid choice. Please try again.")


def main():
    """Main function"""
    print("📄 PDF PROCESSING AND PREDICTION SCRIPT")
    print("=" * 50)
    print("🎯 Features:")
    print("   ✅ Extract blocks from PDFs")
    print("   ✅ Generate heading predictions")
    print("   ✅ Feature alignment and validation")
    print("   ✅ Batch processing of multiple PDFs")
    print("   ✅ Save predictions for manual review")
    print("   ✅ Comprehensive error handling")
    print()
    
    try:
        processor = PDFProcessor()
        processor.interactive_menu()
    except KeyboardInterrupt:
        print("\n👋 Processing interrupted by user")
    except Exception as e:
        logger.error(f"❌ Processing error: {e}")
        raise


if __name__ == "__main__":
    main()
