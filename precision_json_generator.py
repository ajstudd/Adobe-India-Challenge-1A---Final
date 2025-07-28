#!/usr/bin/env python3
"""
Precision-Compatible JSON Generator
==================================

This script generates JSON output using the precision-focused model.
It ensures feature compatibility and applies the same precision filters.

Features:
✅ Compatible with precision-focused model
✅ Applies same feature engineering as training
✅ Uses optimal threshold for precision
✅ Filters out false positives
✅ Generates clean H1, H2, H3 hierarchy

Usage:
    python precision_json_generator.py

Author: AI Assistant
Date: July 27, 2025
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import pickle
import glob
import logging
from pathlib import Path
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
import fitz  # PyMuPDF

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PrecisionJSONGenerator:
    """Generate JSON output with precision-focused model"""
    
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.config = self.load_config()
        
        # Directories
        self.input_dir = os.path.join(self.base_dir, self.config['directories']['input'])
        self.output_dir = os.path.join(self.base_dir, self.config['directories']['output'])
        self.models_dir = os.path.join(self.base_dir, self.config['directories']['models'])
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Model components
        self.model = None
        self.tfidf_vectorizer = None
        self.feature_columns = None
        self.optimal_threshold = None
        
        logger.info("📤 Precision JSON Generator initialized!")
    
    def load_config(self):
        """Load configuration"""
        config_path = os.path.join(self.base_dir, 'config_main.json')
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def load_precision_model(self, version="precision_v1"):
        """Load the precision-focused model"""
        model_path = os.path.join(self.models_dir, f"heading_model_{version}.pkl")
        vectorizer_path = os.path.join(self.models_dir, f"tfidf_vectorizer_{version}.pkl")
        features_path = os.path.join(self.models_dir, f"feature_columns_{version}.pkl")
        threshold_path = os.path.join(self.models_dir, f"optimal_threshold_{version}.pkl")
        
        # Check if all files exist
        required_files = [model_path, vectorizer_path, features_path, threshold_path]
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            logger.error(f"❌ Missing model files: {missing_files}")
            logger.info("💡 Please run precision_focused_trainer.py first")
            return False
        
        try:
            # Load model components
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            with open(vectorizer_path, 'rb') as f:
                self.tfidf_vectorizer = pickle.load(f)
            
            with open(features_path, 'rb') as f:
                self.feature_columns = pickle.load(f)
            
            with open(threshold_path, 'rb') as f:
                self.optimal_threshold = pickle.load(f)
            
            logger.info("✅ Precision model loaded successfully!")
            logger.info(f"🎯 Optimal threshold: {self.optimal_threshold:.3f}")
            logger.info(f"📊 Features: {len(self.feature_columns)}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            return False
    
    def extract_pdf_blocks(self, pdf_path):
        """Extract text blocks from PDF"""
        logger.info(f"📄 Extracting blocks from {os.path.basename(pdf_path)}...")
        
        try:
            doc = fitz.open(pdf_path)
            blocks = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text_blocks = page.get_text("dict")["blocks"]
                
                for block in text_blocks:
                    if "lines" in block:  # Text block
                        for line in block["lines"]:
                            for span in line["spans"]:
                                text = span["text"].strip()
                                if text:
                                    blocks.append({
                                        'text': text,
                                        'font_size': span.get("size", 12.0),
                                        'page': page_num + 1,
                                        'x0': span.get("bbox", [0, 0, 0, 0])[0],
                                        'y0': span.get("bbox", [0, 0, 0, 0])[1],
                                        'x1': span.get("bbox", [0, 0, 0, 0])[2],
                                        'y1': span.get("bbox", [0, 0, 0, 0])[3],
                                        'bold': 'Bold' in span.get("font", ""),
                                        'italic': 'Italic' in span.get("font", ""),
                                        'underline': 0,
                                        'is_all_caps': text.isupper(),
                                        'is_title_case': text.istitle(),
                                        'ends_with_colon': text.endswith(':'),
                                        'starts_with_number': text[0].isdigit() if text else False,
                                        'word_count': len(text.split()),
                                        'relative_font_size': span.get("size", 12.0)
                                    })
            
            doc.close()
            
            if not blocks:
                logger.warning("⚠️  No text blocks extracted")
                return None
            
            df = pd.DataFrame(blocks)
            logger.info(f"✅ Extracted {len(df)} text blocks")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error extracting PDF: {e}")
            return None
    
    def prepare_prediction_features(self, df):
        """Prepare features for prediction (same as training)"""
        logger.info("🔧 Preparing features for prediction...")
        
        # Ensure required columns exist (same as training)
        df = df.copy()
        df['text'] = df['text'].fillna('').astype(str)
        
        # Basic features with defaults
        base_features = ['font_size', 'page', 'x0', 'y0', 'x1', 'y1', 'bold', 'italic', 
                        'underline', 'is_all_caps', 'is_title_case', 'ends_with_colon',
                        'starts_with_number', 'word_count', 'relative_font_size']
        
        for feature in base_features:
            if feature not in df.columns:
                if feature in ['font_size', 'relative_font_size']:
                    df[feature] = 12.0
                elif feature == 'word_count':
                    df[feature] = df['text'].str.split().str.len()
                else:
                    df[feature] = 0
        
        # Convert numeric features
        numeric_features = ['font_size', 'page', 'x0', 'y0', 'x1', 'y1', 'word_count', 'relative_font_size']
        for feature in numeric_features:
            if feature in df.columns:
                df[feature] = pd.to_numeric(df[feature], errors='coerce').fillna(0)
        
        # Enhanced text features (same as training)
        df['text_length'] = df['text'].str.len()
        df['word_count_calc'] = df['text'].str.split().str.len()
        df['avg_word_length'] = df['text'].str.split().apply(lambda x: np.mean([len(w) for w in x]) if x else 0)
        
        # Heading-specific pattern features (same as training)
        df['is_title_case_strict'] = df['text'].str.istitle().astype(int)
        df['is_all_caps_strict'] = df['text'].str.isupper().astype(int)
        df['starts_with_capital'] = df['text'].str.match(r'^[A-Z]', na=False).astype(int)
        df['starts_with_number'] = df['text'].str.match(r'^\d', na=False).astype(int)
        df['ends_with_colon'] = df['text'].str.endswith(':', na=False).astype(int)
        df['has_common_heading_words'] = df['text'].str.contains(
            r'\\b(?:chapter|section|part|introduction|conclusion|summary|abstract|references|appendix|table|figure)\\b', 
            case=False, na=False
        ).astype(int)
        
        # Length-based features (same as training)
        df['is_very_short'] = (df['word_count_calc'] <= 3).astype(int)
        df['is_short'] = (df['word_count_calc'].between(4, 8)).astype(int)
        df['is_medium'] = (df['word_count_calc'].between(9, 15)).astype(int)
        df['is_long'] = (df['word_count_calc'] > 15).astype(int)
        
        # Font features (same as training)
        if 'font_size' in df.columns:
            font_percentiles = self.config['feature_engineering']['font_percentiles']
            for p in font_percentiles:
                thresh = np.percentile(df['font_size'].dropna(), p)
                df[f'font_size_ge_{p}'] = (df['font_size'] >= thresh).astype(int)
        
        # Position features (same as training)
        if 'x0' in df.columns and 'y0' in df.columns:
            df['width'] = (df['x1'] - df['x0']).fillna(0)
            df['height'] = (df['y1'] - df['y0']).fillna(0)
            df['aspect_ratio'] = (df['width'] / (df['height'] + 1)).fillna(0)
            df['is_left_aligned'] = (df['x0'] <= df['x0'].quantile(0.25)).astype(int)
        
        # TF-IDF features using existing vectorizer
        logger.info("📝 Computing TF-IDF features...")
        tfidf_matrix = self.tfidf_vectorizer.transform(df['text'])
        tfidf_feature_names = [f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])]
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_feature_names, index=df.index)
        
        # Combine all features
        feature_df = pd.concat([df, tfidf_df], axis=1)
        
        # Select only the features that were used in training
        available_features = [col for col in self.feature_columns if col in feature_df.columns]
        missing_features = [col for col in self.feature_columns if col not in feature_df.columns]
        
        if missing_features:
            logger.warning(f"⚠️  Missing features: {missing_features}")
            # Add missing features with default values
            for feature in missing_features:
                feature_df[feature] = 0
        
        # Ensure exact feature order
        X = feature_df[self.feature_columns]
        
        logger.info(f"✅ Prepared {len(self.feature_columns)} features for prediction")
        
        return X, feature_df
    
    def predict_headings(self, df):
        """Predict headings using precision model"""
        logger.info("🎯 Predicting headings with precision model...")
        
        if self.model is None:
            logger.error("❌ No model loaded")
            return None
        
        # Prepare features
        X, feature_df = self.prepare_prediction_features(df)
        
        # Make predictions
        y_proba = self.model.predict_proba(X)[:, 1]
        y_pred = (y_proba >= self.optimal_threshold).astype(int)
        
        # Add predictions to dataframe
        result_df = feature_df.copy()
        result_df['heading_probability'] = y_proba
        result_df['is_heading'] = y_pred
        
        # Filter to only predicted headings
        headings_df = result_df[result_df['is_heading'] == 1].copy()
        
        logger.info(f"🎯 Predicted {len(headings_df)} headings out of {len(df)} blocks ({len(headings_df)/len(df)*100:.1f}%)")
        
        return headings_df
    
    def determine_heading_levels(self, headings_df):
        """Determine heading levels based on font size and position"""
        if len(headings_df) == 0:
            return headings_df
        
        # Sort by font size (descending) and page/position
        headings_df = headings_df.sort_values(['font_size', 'page', 'y0'], ascending=[False, True, True])
        
        # Determine levels based on font size percentiles
        font_sizes = headings_df['font_size'].values
        unique_sizes = sorted(set(font_sizes), reverse=True)
        
        if len(unique_sizes) >= 3:
            # Use top 3 font sizes for H1, H2, H3
            h1_size = unique_sizes[0]
            h2_size = unique_sizes[1]
            h3_size = unique_sizes[2]
        elif len(unique_sizes) == 2:
            # Use top 2 sizes
            h1_size = unique_sizes[0]
            h2_size = unique_sizes[1]
            h3_size = unique_sizes[1]  # H3 same as H2
        else:
            # All same size
            h1_size = h2_size = h3_size = unique_sizes[0]
        
        # Assign levels
        headings_df['heading_level'] = 'H3'  # Default
        headings_df.loc[headings_df['font_size'] >= h1_size, 'heading_level'] = 'H1'
        headings_df.loc[(headings_df['font_size'] >= h2_size) & (headings_df['font_size'] < h1_size), 'heading_level'] = 'H2'
        
        # Sort by page and position for final output
        headings_df = headings_df.sort_values(['page', 'y0'])
        
        logger.info(f"📊 Heading levels: {headings_df['heading_level'].value_counts().to_dict()}")
        
        return headings_df
    
    def create_json_output(self, pdf_name, headings_df):
        """Create JSON output in required format"""
        # Extract document title (use first H1 or filename)
        h1_headings = headings_df[headings_df['heading_level'] == 'H1']
        if len(h1_headings) > 0:
            title = h1_headings.iloc[0]['text']
        else:
            title = pdf_name.replace('.pdf', '').replace('_', ' ').title()
        
        # Create outline
        outline = []
        for _, row in headings_df.iterrows():
            outline.append({
                "level": row['heading_level'],
                "text": row['text'],
                "page": int(row['page'])
            })
        
        json_output = {
            "title": title,
            "outline": outline
        }
        
        return json_output
    
    def process_pdf(self, pdf_path):
        """Process a single PDF and generate JSON output"""
        pdf_name = os.path.basename(pdf_path)
        logger.info(f"🚀 Processing {pdf_name}...")
        
        # Extract blocks
        blocks_df = self.extract_pdf_blocks(pdf_path)
        if blocks_df is None:
            logger.error(f"❌ Failed to extract blocks from {pdf_name}")
            return None
        
        # Predict headings
        headings_df = self.predict_headings(blocks_df)
        if headings_df is None or len(headings_df) == 0:
            logger.warning(f"⚠️  No headings predicted for {pdf_name}")
            # Create empty output
            json_output = {
                "title": pdf_name.replace('.pdf', '').replace('_', ' ').title(),
                "outline": []
            }
        else:
            # Determine levels and create output
            headings_df = self.determine_heading_levels(headings_df)
            json_output = self.create_json_output(pdf_name, headings_df)
        
        # Save JSON output
        output_filename = pdf_name.replace('.pdf', '.json')
        output_path = os.path.join(self.output_dir, output_filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_output, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ JSON saved: {output_filename}")
        logger.info(f"📊 Found {len(json_output['outline'])} headings")
        
        return output_path
    
    def process_all_pdfs(self):
        """Process all PDFs in input directory"""
        logger.info("🚀 PROCESSING ALL PDFs IN INPUT DIRECTORY")
        logger.info("=" * 60)
        
        # Find PDF files
        pdf_files = list(Path(self.input_dir).glob("*.pdf"))
        
        if not pdf_files:
            logger.error(f"❌ No PDF files found in {self.input_dir}")
            return False
        
        logger.info(f"📄 Found {len(pdf_files)} PDF files")
        
        # Process each PDF
        success_count = 0
        for pdf_path in pdf_files:
            try:
                result = self.process_pdf(pdf_path)
                if result:
                    success_count += 1
            except Exception as e:
                logger.error(f"❌ Error processing {pdf_path.name}: {e}")
        
        logger.info(f"✅ Successfully processed {success_count}/{len(pdf_files)} PDFs")
        logger.info(f"📁 Output directory: {self.output_dir}")
        
        return success_count > 0

def main():
    """Main function"""
    print("🎯 PRECISION-COMPATIBLE JSON GENERATOR")
    print("=" * 60)
    print("📤 Generates competition JSON output with precision model")
    print("🎯 Focuses on H1, H2, H3 headings with minimal false positives")
    print()
    
    try:
        generator = PrecisionJSONGenerator()
        
        # Load precision model
        if not generator.load_precision_model():
            print("❌ Failed to load precision model")
            print("💡 Please run precision_focused_trainer.py first")
            return
        
        # Process PDFs
        success = generator.process_all_pdfs()
        
        if success:
            print("\\n✅ JSON generation completed successfully!")
            print("📁 Check the output/ directory for generated JSON files")
        else:
            print("\\n❌ JSON generation failed")
            
    except KeyboardInterrupt:
        print("\\n🛑 Process interrupted by user")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        print(f"\\n❌ Error: {e}")

if __name__ == "__main__":
    main()
