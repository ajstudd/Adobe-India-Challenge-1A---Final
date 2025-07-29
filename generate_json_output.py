#!/usr/bin/env python3
"""
JSON Output Generation Script
============================

This script handles generating final JSON output for competition submission.

Features:
✅ Process PDFs from input or unprocessed_pdfs folders
✅ Extract blocks and generate heading predictions
✅ Create structured JSON output with heading hierarchy
✅ Save JSON files to output folder
✅ Validate JSON schema compliance
✅ Handle multilingual documents
✅ Feature compatibility checking
✅ Smart model selection - automatically finds latest retrained model
✅ Display model information during processing
✅ Enhanced model management and status reporting

Model Selection:
- "latest" automatically selects the most recent retrained model
- Prioritizes retrained models over original trained models
- Shows model version, file path, and creation time
- Lists available models for easy selection

Usage:
    python generate_json_output.py

Author: AI Assistant
Date: July 28, 2025 (Enhanced with smart model selection)
"""

import os
import sys
import re
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

# Import intelligent filtering system
try:
    from intelligent_filter import IntelligentFilter
    INTELLIGENT_FILTER_AVAILABLE = True
    logger.info("✅ Intelligent filtering system imported successfully")
except ImportError as e:
    INTELLIGENT_FILTER_AVAILABLE = False
    logger.warning(f"⚠️  Intelligent filtering system not available: {e}")

class JSONOutputGenerator:
    """Generate final JSON output for competition submission"""
    
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load configuration
        self.config = self.load_config()
        
        # Core directories
        self.input_dir = os.path.join(self.base_dir, self.config['directories']['input'])
        self.unprocessed_pdfs_dir = os.path.join(self.base_dir, self.config['directories']['unprocessed_pdfs'])
        self.output_dir = os.path.join(self.base_dir, self.config['directories']['output'])
        self.models_dir = os.path.join(self.base_dir, self.config['directories']['models'])
        
        # Create directories
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Model components
        self.model = None
        self.tfidf_vectorizer = None
        self.feature_columns = None
        self.optimal_threshold = self.config['pipeline_settings']['confidence_threshold']
        self.loaded_model_version = None
        self.loaded_model_path = None
        
        # Feature alignment attributes
        self.feature_mismatch = False
        self.expected_feature_names = None
        self.actual_feature_names = None
        
        logger.info("📤 JSON Output Generator initialized!")
        logger.info(f"📁 Input PDFs: {self.input_dir}")
        logger.info(f"📁 Unprocessed PDFs: {self.unprocessed_pdfs_dir}")
        logger.info(f"📁 Output: {self.output_dir}")
    
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
    
    def find_latest_model(self):
        """Find the most recent model file based on creation time and naming"""
        logger.info("🔍 Finding the most recent model...")
        
        # Get all model files
        model_files = glob.glob(os.path.join(self.models_dir, "heading_model_*.pkl"))
        
        if not model_files:
            logger.error("❌ No model files found!")
            return None
        
        # Priority order: retrained models > trained models > others
        latest_model = None
        latest_time = 0
        
        for model_file in model_files:
            try:
                # Get file modification time
                file_time = os.path.getmtime(model_file)
                filename = os.path.basename(model_file)
                
                # Extract version from filename
                version = filename.replace("heading_model_", "").replace(".pkl", "")
                
                # Prioritize retrained models over others
                priority_bonus = 0
                if "retrained" in version:
                    priority_bonus = 1000000  # High priority for retrained models
                elif "trained" in version and version != "latest":
                    priority_bonus = 100000   # Medium priority for explicitly trained models
                
                effective_time = file_time + priority_bonus
                
                if effective_time > latest_time:
                    latest_time = effective_time
                    latest_model = version
                    
                logger.info(f"   📄 Found: {filename} (time: {file_time}, priority: {priority_bonus})")
                
            except Exception as e:
                logger.warning(f"⚠️  Error checking {model_file}: {e}")
        
        if latest_model:
            logger.info(f"🎯 Latest model determined: {latest_model}")
        else:
            logger.warning("⚠️  Could not determine latest model, falling back to 'latest'")
            latest_model = "latest"
            
        return latest_model

    def load_model(self, version="latest"):
        """Load a saved model"""
        # If version is "latest", find the actual latest model
        if version == "latest":
            actual_version = self.find_latest_model()
            if actual_version and actual_version != "latest":
                logger.info(f"🔄 'latest' resolved to: {actual_version}")
                version = actual_version
        
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
            
            # Store the loaded model version for display
            self.loaded_model_version = version
            self.loaded_model_path = model_path
            
            logger.info(f"✅ Model loaded: {os.path.basename(model_path)}")
            logger.info(f"✅ TF-IDF vectorizer loaded: {os.path.basename(vectorizer_path)}")
            logger.info(f"🏷️  Model version: {version}")
            logger.info(f"🔧 Expected features: {len(self.feature_columns)}")
            logger.info(f"🔧 Optimal threshold: {self.optimal_threshold}")
            return True
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            # Reset everything if loading fails
            self.model = None
            self.tfidf_vectorizer = None
            self.feature_columns = None
            self.loaded_model_version = None
            self.loaded_model_path = None
            return False
    
    def check_feature_compatibility(self):
        """Check feature compatibility by testing with sample data"""
        try:
            # Create a small test sample to check feature compatibility
            test_sample = pd.DataFrame({
                'text': ['Test heading sample'],
                'font_size': [12],
                'page_num': [1],
                'x0': [0], 'y0': [0], 'x1': [100], 'y1': [20]
            })
            test_features = self.prepare_features(test_sample)
            
            expected_features = len(self.feature_columns)
            actual_features = len(test_features.columns)
            
            if actual_features != expected_features:
                logger.warning("⚠️  FEATURE MISMATCH DETECTED!")
                logger.warning(f"   Current config generates: {actual_features} features")
                logger.warning(f"   Loaded model expects: {expected_features} features")
                logger.warning("   Attempting automatic feature alignment...")
                
                # Store the feature alignment info for later use
                self.feature_mismatch = True
                self.expected_feature_names = self.feature_columns
                self.actual_feature_names = list(test_features.columns)
                
                # Find common features
                common_features = [f for f in self.expected_feature_names if f in self.actual_feature_names]
                missing_features = [f for f in self.expected_feature_names if f not in self.actual_feature_names]
                extra_features = [f for f in self.actual_feature_names if f not in self.expected_feature_names]
                
                logger.info(f"✅ Common features: {len(common_features)}")
                if missing_features:
                    logger.warning(f"⚠️  Missing features: {len(missing_features)} - will be filled with zeros")
                    logger.debug(f"   Missing: {missing_features[:5]}{'...' if len(missing_features) > 5 else ''}")
                if extra_features:
                    logger.warning(f"⚠️  Extra features: {len(extra_features)} - will be ignored")
                    logger.debug(f"   Extra: {extra_features[:5]}{'...' if len(extra_features) > 5 else ''}")
                
                logger.info("✅ Feature alignment configured - predictions will work with automatic adjustment")
                return True
            else:
                logger.info("✅ Feature compatibility check passed")
                self.feature_mismatch = False
                return True
                    
        except Exception as e:
            logger.error(f"❌ Error checking feature compatibility: {e}")
            return False
    
    def extract_pdf_to_blocks(self, pdf_path):
        """Extract blocks from PDF using available methods"""
        logger.info(f"📄 Extracting blocks from {os.path.basename(pdf_path)}...")
        
        # Try to use existing extraction scripts
        try:
            sys.path.insert(0, os.path.join(self.base_dir, 'src', 'extraction'))
            from extract_local_dataset_to_csv import process_pdf
            
            output_csv = os.path.join(self.output_dir, f"{Path(pdf_path).stem}_extracted.csv")
            result = process_pdf(pdf_path, output_csv)
            
            if os.path.exists(output_csv):
                df = pd.read_csv(output_csv)
                # Clean up temporary file
                os.remove(output_csv)
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
                distances.append(0)  # Default to 0 instead of None for prediction
        
        df[distance_col] = distances
        return df
    
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
            
            # More granular font size percentiles - MUST MATCH training configuration
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
            # Use the same percentiles as in training configuration
            font_percentiles = self.config.get('feature_engineering', {}).get('font_percentiles', [50, 75, 90, 95])
            for p in font_percentiles:
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
        
        # Add original CSV features if available (CRITICAL: needed for retrained model compatibility)
        csv_features = ['distance_to_previous_heading', 'line_spacing_above', 'relative_font_size', 
                       'line_position_on_page', 'bold', 'italic', 'underline']
        
        for feature in csv_features:
            if feature in df.columns:
                # Use actual CSV values
                features[feature] = df[feature].fillna(0)
                logger.debug(f"✅ Using CSV feature: {feature}")
            else:
                # Provide sensible defaults that match retrained model expectations
                if feature == 'distance_to_previous_heading':
                    features[feature] = 0  # Will be calculated later if needed
                elif feature == 'line_spacing_above':
                    features[feature] = 0
                elif feature == 'relative_font_size':
                    features[feature] = 1.0
                elif feature == 'line_position_on_page':
                    features[feature] = 0.5
                elif feature in ['bold', 'italic', 'underline']:
                    features[feature] = 0
                logger.debug(f"➕ Added default for missing feature: {feature}")
        
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
        
        # Calculate distance_to_previous_heading if it's not in original data or is all zeros
        if 'distance_to_previous_heading' not in df.columns or df['distance_to_previous_heading'].fillna(0).sum() == 0:
            logger.info("📏 Calculating distance_to_previous_heading for prediction compatibility...")
            # Create a temporary DataFrame with dummy headings (all zeros for prediction)
            temp_df = df.copy()
            temp_df['is_heading'] = 0  # Dummy values for distance calculation
            temp_df = self._calculate_distance_to_previous_heading(temp_df)
            features_df['distance_to_previous_heading'] = temp_df['distance_to_previous_heading'].fillna(0)
        
        # Add POS-based features if enabled and not already present
        pos_features = ['num_nouns', 'num_verbs', 'num_adjs', 'num_advs', 'num_propn', 'num_pronouns', 'num_other_pos']
        pos_features_enabled = self.config.get('feature_engineering', {}).get('pos_features', False)
        
        # Check if POS features already exist in the original data
        pos_features_exist = all(col in df.columns for col in pos_features)
        if pos_features_exist:
            logger.info("📄 Using existing POS features from input data...")
            for feature in pos_features:
                features_df[feature] = df[feature].fillna(0)
        elif pos_features_enabled:
            logger.info("🏷️  Computing POS features (not found in input data)...")
            features_df = self._add_pos_features(df, features_df)
        else:
            # Add zero-filled POS features for consistency
            logger.info("🏷️  Adding zero-filled POS features (disabled in config)...")
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
    
    def predict_headings(self, df):
        """Predict headings for a DataFrame of blocks"""
        if self.model is None:
            logger.error("❌ No model loaded. Load a model first.")
            return None
        
        if self.tfidf_vectorizer is None:
            logger.error("❌ TF-IDF vectorizer not loaded. Cannot proceed with prediction.")
            return None

        # Prepare features (using existing vectorizer)
        X = self.prepare_features(df)
        
        # Enhanced feature alignment - always align to ensure consistency
        logger.info(f"🔧 Aligning features: {len(X.columns)} → {len(self.feature_columns)}")
        
        # Create columns data for efficient DataFrame construction
        aligned_columns = {}
        missing_count = 0
        
        for col in self.feature_columns:
            if col in X.columns:
                aligned_columns[col] = X[col]
            else:
                # Use appropriate default values based on feature type
                if 'tfidf_' in col:
                    default_value = 0.0
                elif any(keyword in col.lower() for keyword in ['ratio', 'norm', 'relative']):
                    default_value = 0.0
                elif any(keyword in col.lower() for keyword in ['count', 'length', 'size', 'position']):
                    default_value = 0.0
                elif any(keyword in col.lower() for keyword in ['is_', 'has_', 'starts_', 'ends_']):
                    default_value = 0
                else:
                    default_value = 0.0
                
                aligned_columns[col] = pd.Series([default_value] * len(X), index=X.index)
                missing_count += 1
        
        # Create aligned DataFrame efficiently using pd.concat
        aligned_X = pd.DataFrame(aligned_columns, index=X.index)
        
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
        result_df['is_heading_pred'] = predictions  # Store original ML predictions
        result_df['heading_confidence'] = probabilities
        result_df['predicted_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Apply intelligent filtering if available
        if INTELLIGENT_FILTER_AVAILABLE:
            logger.info("🧠 Applying intelligent rule-based filtering...")
            try:
                # Initialize intelligent filter
                intelligent_filter = IntelligentFilter(config_path=os.path.join(self.base_dir, 'config_main.json'))
                
                # Apply filtering
                filtered_df = intelligent_filter.apply_intelligent_filtering(result_df, confidence_col='heading_confidence')
                
                # Generate filtering report with error handling
                try:
                    report_path = os.path.join(self.base_dir, f"filtering_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                    report = intelligent_filter.generate_filtering_report(filtered_df, report_path)
                    logger.info(f"   📋 Filtering report: {report_path}")
                except Exception as report_error:
                    logger.warning(f"⚠️  Error generating filtering report: {report_error}")
                    logger.info("   📋 Filtering report generation failed, but filtering succeeded")
                
                # Update final predictions
                result_df['is_heading'] = filtered_df['is_heading_pred']
                result_df['filter_score'] = filtered_df.get('filter_score', 0.0)
                result_df['filter_decision'] = filtered_df.get('filter_decision', 'not_filtered')
                result_df['filter_reasons'] = filtered_df.get('filter_reasons', '')
                
                # Log filtering results
                original_count = predictions.sum()
                final_count = result_df['is_heading'].sum()
                filtered_count = original_count - final_count
                
                logger.info(f"📊 Intelligent Filtering Results:")
                logger.info(f"   🔢 Original ML predictions: {original_count}")
                logger.info(f"   ✅ Final filtered predictions: {final_count}")
                logger.info(f"   ❌ Filtered out: {filtered_count} ({(filtered_count/original_count*100):.1f}%)")
                
            except Exception as e:
                logger.error(f"❌ Error in intelligent filtering: {e}")
                logger.warning("⚠️  Falling back to original ML predictions")
                result_df['is_heading'] = predictions
                result_df['filter_score'] = 0.0
                result_df['filter_decision'] = 'filter_error'
        else:
            logger.warning("⚠️  Intelligent filtering not available, using original ML predictions")
            result_df['is_heading'] = predictions
        
        return result_df
    
    def determine_heading_level(self, row, font_percentiles):
        """Determine heading level (H1/H2/H3) based on font size, structure, and metadata"""
        font_size = row.get('font_size', 12)
        y_position = row.get('y0', row.get('y_position', 0))
        page_num = row.get('page_num', row.get('page', 1))
        text = row.get('text', '').strip()
        
        # Get font thresholds
        high_font = font_percentiles.get(90, 16)
        medium_font = font_percentiles.get(75, 14)
        low_font = font_percentiles.get(60, 12)
        
        # Enhanced metadata-based level detection
        
        # H1 Level Indicators (Main sections/chapters)
        h1_patterns = [
            r'^\s*(?:chapter|part)\s+\d+',  # Chapter 1, Part 1
            r'^\s*[IVX]+\.\s*[A-Z]',  # I.Introduction, II.Methodology  
            r'^\s*\d+\.\s*[A-Z][a-z]*(?:\s+[A-Z][a-z]*)*$',  # 1.Introduction, 2.System Architecture
            r'^\s*(?:introduction|conclusion|summary|abstract|references|bibliography|acknowledgments?|methodology|results|discussion|overview|background)\s*$',
        ]
        
        # H2 Level Indicators (Subsections)
        h2_patterns = [
            r'^\s*\d+\.\d+\s+[A-Z]',  # 1.1 Something, 2.3 Another
            r'^\s*[A-Z]\.\s+[A-Z]',  # A. Something, B. Another
            r'^\s*\d+\.\d+\.\s*[A-Z][a-z]*(?:\s+[A-Z][a-z]*)*$',  # 1.1.Introduction
        ]
        
        # H3 Level Indicators (Sub-subsections)
        h3_patterns = [
            r'^\s*\d+\.\d+\.\d+\s+[A-Z]',  # 1.1.1 Something
            r'^\s*[a-z][\.)]\s+[A-Z]',  # a) Something, b. Another
            r'^\s*\([a-z]\)\s+[A-Z]',  # (a) Something
        ]
        
        # Check structural patterns first (highest priority)
        for pattern in h1_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return "H1"
                
        for pattern in h2_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return "H2"
                
        for pattern in h3_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return "H3"
        
        # Font-based classification with enhanced thresholds
        if font_size >= high_font:
            # Large fonts are typically H1
            return "H1"
        elif font_size >= medium_font:
            # Medium fonts could be H1 or H2 depending on context
            # If it's at the start of a page or has certain patterns, make it H1
            if (page_num == 1 and y_position < 300) or font_size >= font_percentiles.get(95, 18):
                return "H1"
            else:
                return "H2"
        elif font_size >= low_font:
            # Smaller fonts are typically H2 or H3
            # Check if it looks like a main section
            if re.match(r'^\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*$', text) and len(text.split()) <= 4:
                return "H2"
            else:
                return "H3"
        else:
            # Very small fonts are H3
            return "H3"
    
    def create_json_structure(self, pdf_name, headings_df, all_blocks_df=None):
        """Create JSON structure for a PDF"""
        
        # Find document title from first H1 in initial 100 blocks
        document_title = pdf_name.replace('_', ' ').title()  # Default fallback
        
        if all_blocks_df is not None:
            # Look for first H1 heading in initial 100 blocks
            initial_blocks = all_blocks_df.head(100)
            initial_headings = initial_blocks[initial_blocks.get('is_heading', 0) == 1]
            
            if len(initial_headings) > 0:
                # Check each heading to see if it's H1 level
                for _, row in initial_headings.iterrows():
                    # Use intelligent filter's heading level if available
                    if 'final_heading_level' in row and row['final_heading_level'] == 'H1':
                        document_title = str(row['text']).strip()
                        logger.info(f"📝 Found document title from initial blocks: '{document_title}'")
                        break
                    else:
                        # Determine level using existing logic
                        font_percentiles = {75: 14, 90: 16, 95: 18}
                        if 'font_size' in all_blocks_df.columns and len(all_blocks_df) > 0:
                            for p in [75, 90, 95]:
                                font_percentiles[p] = all_blocks_df['font_size'].quantile(p / 100.0)
                        
                        level = self.determine_heading_level(row, font_percentiles)
                        if level == 'H1':
                            document_title = str(row['text']).strip()
                            logger.info(f"📝 Found document title from initial blocks: '{document_title}'")
                            break
        
        # Calculate font percentiles for heading level determination (fallback)
        if 'font_size' in headings_df.columns and len(headings_df) > 0:
            font_percentiles = {}
            for p in [75, 90, 95]:
                font_percentiles[p] = headings_df['font_size'].quantile(p / 100.0)
        else:
            font_percentiles = {75: 14, 90: 16, 95: 18}
        
        # Create outline entries with enhanced heading level detection
        outline = []
        for _, row in headings_df.iterrows():
            # Use intelligent filter's heading level if available
            if 'final_heading_level' in row and row['final_heading_level'] in ['H1', 'H2', 'H3']:
                level = row['final_heading_level']
            else:
                # Fallback to traditional method
                level = self.determine_heading_level(row, font_percentiles)
            
            outline.append({
                "level": level,
                "text": str(row['text']).strip(),
                "page": int(row.get('page_num', row.get('page', 1)))
            })
        
        # Sort by page number and y-position if available
        if 'y0' in headings_df.columns:
            # Check which page column is available
            page_col = None
            if 'page_num' in headings_df.columns:
                page_col = 'page_num'
            elif 'page' in headings_df.columns:
                page_col = 'page'
            
            # Sort by page and y-position if page column is available
            if page_col:
                headings_df_sorted = headings_df.sort_values([page_col, 'y0'])
            else:
                # Fallback: sort by y0 only
                headings_df_sorted = headings_df.sort_values(['y0'])
            
            outline = []
            for _, row in headings_df_sorted.iterrows():
                # Use intelligent filter's heading level if available
                if 'final_heading_level' in row and row['final_heading_level'] in ['H1', 'H2', 'H3']:
                    level = row['final_heading_level']
                else:
                    # Fallback to traditional method
                    level = self.determine_heading_level(row, font_percentiles)
                    
                outline.append({
                    "level": level,
                    "text": str(row['text']).strip(),
                    "page": int(row.get('page_num', row.get('page', 1)))
                })
        
        # Create final JSON structure
        json_output = {
            "title": document_title,
            "outline": outline
        }
        
        # Apply final consecutive heading merging
        json_output = self.merge_consecutive_headings_in_outline(json_output)
        
        return json_output
    
    def merge_consecutive_headings_in_outline(self, json_output):
        """
        Merge consecutive headings in the final outline where the first heading 
        matches mergeable patterns (numbers, roman numerals, letters with dots)
        """
        if 'outline' not in json_output or not json_output['outline']:
            return json_output
        
        import re
        
        # Define mergeable prefix patterns
        mergeable_patterns = {
            'numbers': re.compile(r'^\s*\d+\s*$'),                       # "1", "2", etc.
            'numbered_dots': re.compile(r'^\s*\d+\.\s*$'),               # "1.", "2.", etc.
            'numbered_multi': re.compile(r'^\s*\d+\.\d+\s*$'),           # "1.1", "2.3", etc.
            'roman_numerals': re.compile(r'^\s*[IVX]+\s*$', re.IGNORECASE),     # "I", "II", "VII", etc.
            'roman_numerals_dots': re.compile(r'^\s*[IVX]+\.\s*$', re.IGNORECASE),  # "I.", "VII.", etc.
            'single_letter': re.compile(r'^\s*[A-Z]\.\s*$'),             # "A.", "B.", etc.
            'letter_no_dot': re.compile(r'^\s*[A-Z]\s*$'),               # "A", "B", etc.
        }
        
        outline = json_output['outline']
        merged_outline = []
        i = 0
        merged_count = 0
        
        logger.info("🔗 Applying final consecutive heading merge to outline...")
        
        while i < len(outline):
            current_entry = outline[i]
            current_text = current_entry['text'].strip()
            
            # Check if current text matches any mergeable pattern
            is_mergeable = False
            pattern_name = ""
            
            for name, pattern in mergeable_patterns.items():
                if pattern.match(current_text):
                    is_mergeable = True
                    pattern_name = name
                    break
            
            # If current heading is mergeable and there's a next heading on the same page
            if (is_mergeable and 
                i + 1 < len(outline) and
                outline[i + 1]['page'] == current_entry['page']):
                
                next_entry = outline[i + 1]
                next_text = next_entry['text'].strip()
                
                # Additional validation: next text should look like a proper heading continuation
                if (len(next_text) > 0 and 
                    next_text[0].isupper() and  # Starts with uppercase
                    len(next_text.split()) <= 10):  # Not too long
                    
                    # Merge the headings
                    merged_text = f"{current_text} {next_text}".strip()
                    merged_entry = {
                        "level": next_entry['level'],  # Use the level of the second heading
                        "text": merged_text,
                        "page": current_entry['page']
                    }
                    
                    merged_outline.append(merged_entry)
                    merged_count += 1
                    logger.debug(f"Merged outline headings: '{current_text}' + '{next_text}' -> '{merged_text}'")
                    
                    # Skip the next entry since we merged it
                    i += 2
                    continue
            
            # Also check for broken headings (consecutive single words)
            if (len(current_text.split()) == 1 and 
                current_text.isalpha() and 
                current_text[0].isupper() and
                i + 1 < len(outline) and
                outline[i + 1]['page'] == current_entry['page']):
                
                # Look ahead for more single words to merge
                words_to_merge = [current_text]
                entries_to_merge = [current_entry]
                j = i + 1
                
                while (j < len(outline) and 
                       j < i + 4 and  # Look at most 3 entries ahead
                       outline[j]['page'] == current_entry['page']):
                    
                    next_text = outline[j]['text'].strip()
                    
                    # Include single words or short phrases
                    if (len(next_text.split()) <= 2 and 
                        next_text.isalpha() and 
                        next_text[0].isupper()):
                        words_to_merge.append(next_text)
                        entries_to_merge.append(outline[j])
                        j += 1
                    else:
                        break
                
                # Merge if we have multiple words
                if len(words_to_merge) >= 2:
                    merged_text = " ".join(words_to_merge)
                    
                    # Validate the merged text looks reasonable
                    if (len(merged_text) >= 5 and 
                        len(merged_text.split()) <= 6):
                        
                        merged_entry = {
                            "level": current_entry['level'],
                            "text": merged_text,
                            "page": current_entry['page']
                        }
                        
                        merged_outline.append(merged_entry)
                        merged_count += 1
                        logger.debug(f"Merged broken heading: {' + '.join(words_to_merge)} -> '{merged_text}'")
                        
                        # Skip all merged entries
                        i = j
                        continue
            
            # No merge needed, add current entry as-is
            merged_outline.append(current_entry)
            i += 1
        
        logger.info(f"✅ Final outline merge completed: {merged_count} merges performed")
        
        # Update the JSON output
        json_output['outline'] = merged_outline
        return json_output
    
    def validate_json_schema(self, json_data):
        """Validate JSON output against expected schema"""
        try:
            # Check required fields
            if 'title' not in json_data:
                logger.warning("⚠️  Missing 'title' field in JSON")
                return False
            
            if 'outline' not in json_data:
                logger.warning("⚠️  Missing 'outline' field in JSON")
                return False
            
            if not isinstance(json_data['outline'], list):
                logger.warning("⚠️  'outline' should be a list")
                return False
            
            # Check outline entries
            for i, entry in enumerate(json_data['outline']):
                if not isinstance(entry, dict):
                    logger.warning(f"⚠️  Outline entry {i} should be a dictionary")
                    return False
                
                required_fields = ['level', 'text', 'page']
                for field in required_fields:
                    if field not in entry:
                        logger.warning(f"⚠️  Missing '{field}' in outline entry {i}")
                        return False
                
                # Validate level
                if entry['level'] not in ['H1', 'H2', 'H3']:
                    logger.warning(f"⚠️  Invalid level '{entry['level']}' in entry {i}")
                    return False
                
                # Validate page number
                if not isinstance(entry['page'], int) or entry['page'] < 1:
                    logger.warning(f"⚠️  Invalid page number '{entry['page']}' in entry {i}")
                    return False
            
            logger.info("✅ JSON schema validation passed")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️  JSON validation error: {e}")
            return False
    
    def generate_json_output(self, source_dir="input"):
        """Generate final JSON output for competition submission"""
        logger.info("\n" + "="*70)
        logger.info("📤 GENERATING JSON OUTPUT")
        logger.info("="*70)
        
        if self.model is None:
            logger.warning("⚠️  No model loaded. Attempting to load existing model...")
            if not self.load_model():
                logger.error("❌ No model found. Please train or load a model first.")
                return False
        
        if self.tfidf_vectorizer is None:
            logger.warning("⚠️  TF-IDF vectorizer not loaded. Attempting to load...")
            if not self.load_model():
                logger.error("❌ Failed to load model and vectorizer")
                return False
        
        # Display model information
        logger.info(f"🤖 Using model: {self.loaded_model_version or 'Unknown version'}")
        if self.loaded_model_path:
            logger.info(f"📁 Model file: {os.path.basename(self.loaded_model_path)}")
        logger.info(f"🔧 Confidence threshold: {self.optimal_threshold}")
        logger.info(f"📊 Expected features: {len(self.feature_columns) if self.feature_columns else 'Unknown'}")
        
        # Check feature compatibility
        if not self.check_feature_compatibility():
            return False

        # Determine which PDFs to process
        if source_dir == "input" and os.path.exists(self.input_dir):
            pdf_files = list(Path(self.input_dir).glob("*.pdf"))
            source_folder = "input"
        else:
            pdf_files = list(Path(self.unprocessed_pdfs_dir).glob("*.pdf"))
            source_folder = "unprocessed_pdfs"
        
        if not pdf_files:
            logger.warning(f"⚠️  No PDF files found in {source_folder} directory")
            return False
        
        logger.info(f"📄 Processing {len(pdf_files)} PDFs from {source_folder} folder")
        
        successful = 0
        failed = 0
        
        for i, pdf_path in enumerate(pdf_files):
            pdf_name = pdf_path.stem
            logger.info(f"\n--- Generating JSON for PDF {i+1}/{len(pdf_files)}: {pdf_name} ---")
            
            try:
                # Extract and predict
                df_blocks = self.extract_pdf_to_blocks(str(pdf_path))
                
                if df_blocks is None or len(df_blocks) == 0:
                    logger.error(f"❌ Failed to extract blocks from {pdf_name}")
                    failed += 1
                    continue
                
                df_predictions = self.predict_headings(df_blocks)
                
                if df_predictions is None:
                    logger.error(f"❌ Failed to generate predictions for {pdf_name}")
                    failed += 1
                    continue
                
                # Filter predicted headings
                headings_df = df_predictions[df_predictions['is_heading'] == 1].copy()
                
                if len(headings_df) == 0:
                    logger.warning(f"⚠️  No headings detected in {pdf_name}")
                    # Try to find document title from first H1 in initial blocks anyway
                    document_title = pdf_name.replace('_', ' ').title()  # Default fallback
                    
                    if df_predictions is not None:
                        # Look for first H1 heading in initial 100 blocks
                        initial_blocks = df_predictions.head(100)
                        initial_headings = initial_blocks[initial_blocks.get('is_heading', 0) == 1]
                        
                        if len(initial_headings) > 0:
                            # Check each heading to see if it's H1 level
                            for _, row in initial_headings.iterrows():
                                # Use intelligent filter's heading level if available
                                if 'final_heading_level' in row and row['final_heading_level'] == 'H1':
                                    document_title = str(row['text']).strip()
                                    logger.info(f"📝 Found document title from initial blocks: '{document_title}'")
                                    break
                                else:
                                    # Determine level using existing logic
                                    font_percentiles = {75: 14, 90: 16, 95: 18}
                                    if 'font_size' in df_predictions.columns and len(df_predictions) > 0:
                                        for p in [75, 90, 95]:
                                            font_percentiles[p] = df_predictions['font_size'].quantile(p / 100.0)
                                    
                                    level = self.determine_heading_level(row, font_percentiles)
                                    if level == 'H1':
                                        document_title = str(row['text']).strip()
                                        logger.info(f"📝 Found document title from initial blocks: '{document_title}'")
                                        break
                    
                    # Create minimal JSON with no outline
                    json_output = {
                        "title": document_title,
                        "outline": []
                    }
                else:
                    # Create JSON structure
                    json_output = self.create_json_structure(pdf_name, headings_df, df_predictions)
                
                # Validate JSON schema
                if not self.validate_json_schema(json_output):
                    logger.warning(f"⚠️  JSON validation failed for {pdf_name}, but continuing...")
                
                # Save JSON
                json_path = os.path.join(self.output_dir, f"{pdf_name}.json")
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(json_output, f, indent=2, ensure_ascii=False)
                
                logger.info(f"✅ Generated JSON with {len(json_output['outline'])} headings")
                logger.info(f"💾 Saved: {json_path}")
                
                successful += 1
                
            except Exception as e:
                logger.error(f"❌ Error processing {pdf_name}: {e}")
                failed += 1
        
        logger.info(f"\n🎯 JSON generation complete!")
        logger.info(f"✅ Successfully processed: {successful}/{len(pdf_files)} PDFs")
        logger.info(f"❌ Failed: {failed} PDFs")
        logger.info(f"📁 JSON files saved to: {self.output_dir}")
        
        return successful > 0
    
    def show_status(self):
        """Show generator status"""
        print("\n📊 JSON GENERATOR STATUS")
        print("-" * 40)
        
        # Model status
        model_loaded = self.model is not None
        print(f"🤖 Model loaded: {'✅ Yes' if model_loaded else '❌ No'}")
        
        if model_loaded:
            print(f"🏷️  Model version: {self.loaded_model_version or 'Unknown'}")
            if self.loaded_model_path:
                print(f"📁 Model file: {os.path.basename(self.loaded_model_path)}")
            print(f"🔧 Optimal threshold: {self.optimal_threshold}")
            print(f"📊 Expected features: {len(self.feature_columns) if self.feature_columns else 'Unknown'}")
        else:
            print("   💡 Use option 1 to load a model")
        
        # PDF files status
        input_pdfs = list(Path(self.input_dir).glob("*.pdf"))
        print(f"📥 Input PDFs: {len(input_pdfs)}")
        
        unprocessed_pdfs = list(Path(self.unprocessed_pdfs_dir).glob("*.pdf"))
        print(f"📄 Unprocessed PDFs: {len(unprocessed_pdfs)}")
        
        # Output status
        json_files = list(Path(self.output_dir).glob("*.json"))
        print(f"📤 JSON output files: {len(json_files)}")
        
        # Available models status
        model_files = glob.glob(os.path.join(self.models_dir, "heading_model_*.pkl"))
        print(f"🗃️  Available models: {len(model_files)}")
        if model_files:
            print("   📋 Recent models:")
            # Sort by modification time (most recent first)
            model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            for i, model_file in enumerate(model_files[:5]):  # Show top 5
                filename = os.path.basename(model_file)
                version = filename.replace("heading_model_", "").replace(".pkl", "")
                mod_time = datetime.fromtimestamp(os.path.getmtime(model_file))
                print(f"      {i+1}. {version} ({mod_time.strftime('%Y-%m-%d %H:%M')})")
            if len(model_files) > 5:
                print(f"      ... and {len(model_files)-5} more models")
    
    def interactive_menu(self):
        """Interactive menu for JSON generation operations"""
        while True:
            print("\n" + "="*60)
            print("📤 JSON OUTPUT GENERATION SCRIPT")
            print("="*60)
            print("1. 📤 Load model")
            print("2. 📥 Generate JSON from input PDFs")
            print("3. 📄 Generate JSON from unprocessed PDFs")
            print("4. 📊 Show status")
            print("5. 🧹 Clean output directory")
            print("6. ❌ Exit")
            print()
            
            choice = input("Choose an option (1-6): ").strip()
            
            if choice == '1':
                # Show available models first
                model_files = glob.glob(os.path.join(self.models_dir, "heading_model_*.pkl"))
                if model_files:
                    print("\n📋 Available models:")
                    model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                    for i, model_file in enumerate(model_files[:10]):  # Show top 10
                        filename = os.path.basename(model_file)
                        version = filename.replace("heading_model_", "").replace(".pkl", "")
                        mod_time = datetime.fromtimestamp(os.path.getmtime(model_file))
                        
                        # Highlight retrained models
                        prefix = "🔄" if "retrained" in version else "🤖"
                        print(f"   {i+1}. {prefix} {version} ({mod_time.strftime('%Y-%m-%d %H:%M')})")
                    
                    if len(model_files) > 10:
                        print(f"   ... and {len(model_files)-10} more models")
                else:
                    print("\n⚠️  No models found in models directory!")
                
                print("\n💡 Options:")
                print("   - Type 'latest' (recommended) - automatically selects most recent retrained model")
                print("   - Type specific version name (e.g., 'retrained_cycle_2_20250728_103027')")
                print("   - Press Enter for 'latest'")
                
                version = input("\nModel version: ").strip()
                if not version:
                    version = "latest"
                
                if self.load_model(version):
                    print("✅ Model loaded successfully!")
                    if self.check_feature_compatibility():
                        print("✅ Feature compatibility check passed!")
                    else:
                        print("❌ Feature compatibility check failed!")
                else:
                    print("❌ Failed to load model!")
            
            elif choice == '2':
                if self.model is None:
                    print("❌ No model loaded. Please load a model first.")
                    continue
                
                success = self.generate_json_output("input")
                if success:
                    print("✅ JSON generation completed successfully!")
                else:
                    print("❌ JSON generation failed!")
            
            elif choice == '3':
                if self.model is None:
                    print("❌ No model loaded. Please load a model first.")
                    continue
                
                success = self.generate_json_output("unprocessed")
                if success:
                    print("✅ JSON generation completed successfully!")
                else:
                    print("❌ JSON generation failed!")
            
            elif choice == '4':
                self.show_status()
            
            elif choice == '5':
                confirm = input("Delete all JSON output files? (y/N): ").strip().lower()
                if confirm == 'y':
                    import shutil
                    if os.path.exists(self.output_dir):
                        shutil.rmtree(self.output_dir)
                        os.makedirs(self.output_dir)
                        print("✅ Cleaned output directory")
                else:
                    print("❌ Cleanup cancelled")
            
            elif choice == '6':
                print("👋 Goodbye!")
                break
            
            else:
                print("❌ Invalid choice. Please try again.")


def main():
    """Main function"""
    print("📤 JSON OUTPUT GENERATION SCRIPT")
    print("=" * 50)
    print("🎯 Features:")
    print("   ✅ Process PDFs from input or unprocessed folders")
    print("   ✅ Extract blocks and generate heading predictions")
    print("   ✅ Create structured JSON output with hierarchy")
    print("   ✅ Save JSON files to output folder")
    print("   ✅ Validate JSON schema compliance")
    print("   ✅ Handle multilingual documents")
    print("   ✅ Feature compatibility checking")
    print()
    
    try:
        generator = JSONOutputGenerator()
        generator.interactive_menu()
    except KeyboardInterrupt:
        print("\n👋 Generation interrupted by user")
    except Exception as e:
        logger.error(f"❌ Generation error: {e}")
        raise


if __name__ == "__main__":
    main()
