#!/usr/bin/env python3
"""
Enhanced Feature Calculator for Retraining
==========================================

This script calculates missing features in prediction CSV files before retraining.
Specifically handles:
- distance_to_previous_heading calculation
- POS features (if spaCy is available)
- Feature alignment and validation

Usage:
    python enhance_prediction_features.py

Author: AI Assistant
Date: July 28, 2025
"""

import os
import pandas as pd
import numpy as np
import glob
import logging
import subprocess
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_distance_to_previous_heading(df, heading_col='is_heading', distance_col='distance_to_previous_heading'):
    """Calculate distance to previous heading for each row"""
    logger.info("📏 Calculating distance_to_previous_heading...")
    
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
    logger.info(f"✅ Calculated distance_to_previous_heading for {len(df)} rows")
    return df

def calculate_pos_features(df):
    """Calculate POS features if spaCy is available"""
    try:
        import spacy
        from langdetect import detect
        
        # Try to load the spaCy model
        try:
            nlp = spacy.load("en_core_web_sm")
            logger.info("✅ Successfully loaded spaCy English model")
        except OSError:
            logger.warning("⚠️  English spaCy model not found. Installing...")
            try:
                subprocess.check_call([
                    subprocess.sys.executable, "-m", "spacy", "download", "en_core_web_sm"
                ])
                nlp = spacy.load("en_core_web_sm")
                logger.info("✅ Successfully installed and loaded spaCy English model")
            except Exception as e:
                logger.warning(f"⚠️  Could not install spaCy model: {e}")
                return df
        
        logger.info("🏷️  Calculating POS features...")
        
        # Initialize POS feature columns if they don't exist
        pos_features = ['num_nouns', 'num_verbs', 'num_adjs', 'num_advs', 'num_propn', 'num_pronouns', 'num_other_pos']
        for feature in pos_features:
            if feature not in df.columns:
                df[feature] = 0
        
        processed_count = 0
        error_count = 0
        
        # Process text for POS features with progress tracking
        for idx, row in df.iterrows():
            text = row.get('text', '')
            
            # Skip empty or NaN text
            if pd.isna(text) or len(str(text).strip()) == 0:
                continue
                
            try:
                text_str = str(text).strip()
                
                # Quick language detection for longer texts
                lang = 'en'  # Default to English
                if len(text_str) > 10:
                    try:
                        detected_lang = detect(text_str)
                        if detected_lang:
                            lang = detected_lang
                    except:
                        lang = 'en'  # Keep English as fallback
                
                # Process with spaCy (for now, process all text, not just English)
                # This allows better handling of mixed-language documents
                if len(text_str) > 0:
                    doc = nlp(text_str)
                    
                    # Count POS tags efficiently
                    pos_counts = {
                        'num_nouns': 0,
                        'num_verbs': 0, 
                        'num_adjs': 0,
                        'num_advs': 0,
                        'num_propn': 0,
                        'num_pronouns': 0,
                        'num_other_pos': 0
                    }
                    
                    # Count tokens by POS
                    for token in doc:
                        if token.pos_ == 'NOUN':
                            pos_counts['num_nouns'] += 1
                        elif token.pos_ == 'VERB':
                            pos_counts['num_verbs'] += 1
                        elif token.pos_ == 'ADJ':
                            pos_counts['num_adjs'] += 1
                        elif token.pos_ == 'ADV':
                            pos_counts['num_advs'] += 1
                        elif token.pos_ == 'PROPN':
                            pos_counts['num_propn'] += 1
                        elif token.pos_ == 'PRON':
                            pos_counts['num_pronouns'] += 1
                        elif token.pos_ not in ['SPACE', 'PUNCT']:
                            pos_counts['num_other_pos'] += 1
                    
                    # Update dataframe with counts
                    for feature, count in pos_counts.items():
                        df.at[idx, feature] = count
                    
                    processed_count += 1
                    
                    # Progress logging every 100 rows
                    if processed_count % 100 == 0:
                        logger.info(f"   📝 Processed {processed_count} rows...")
                        
            except Exception as e:
                # Log error but continue processing
                error_count += 1
                logger.debug(f"Skipping POS for row {idx}: {e}")
                continue
        
        logger.info(f"✅ POS features calculated for {processed_count} rows")
        if error_count > 0:
            logger.warning(f"⚠️  Skipped {error_count} rows due to processing errors")
        
        return df
        
    except ImportError as ie:
        logger.warning(f"⚠️  spaCy/langdetect not available: {ie}")
        logger.warning("   Installing fallback zero-filled POS features")
        # Add zero-filled POS features as fallback
        pos_features = ['num_nouns', 'num_verbs', 'num_adjs', 'num_advs', 'num_propn', 'num_pronouns', 'num_other_pos']
        for feature in pos_features:
            if feature not in df.columns:
                df[feature] = 0
        return df

def enhance_prediction_file(csv_path):
    """Enhance a single prediction CSV file with missing features"""
    logger.info(f"🔧 Enhancing {os.path.basename(csv_path)}...")
    
    try:
        # Load the CSV
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        logger.info(f"📄 Loaded {len(df)} rows from {os.path.basename(csv_path)}")
        
        original_columns = list(df.columns)
        
        # Calculate distance_to_previous_heading if missing or empty
        if 'distance_to_previous_heading' not in df.columns or df['distance_to_previous_heading'].isna().all():
            df = calculate_distance_to_previous_heading(df)
        elif df['distance_to_previous_heading'].isna().any():
            # Recalculate if partially missing
            logger.info("📏 Recalculating distance_to_previous_heading (partially missing)...")
            df = calculate_distance_to_previous_heading(df)
        
        # Calculate POS features if missing or all zeros
        pos_features = ['num_nouns', 'num_verbs', 'num_adjs', 'num_advs', 'num_propn', 'num_pronouns', 'num_other_pos']
        missing_pos_features = [f for f in pos_features if f not in df.columns]
        
        # Check if POS features need to be calculated
        needs_pos_calculation = False
        
        if missing_pos_features:
            logger.info(f"🏷️  Missing POS features: {missing_pos_features}")
            needs_pos_calculation = True
        else:
            # Check if all POS features are zeros (indicating they weren't calculated properly)
            all_zeros = True
            for feature in pos_features:
                if df[feature].sum() > 0:  # If any feature has non-zero values
                    all_zeros = False
                    break
            
            if all_zeros:
                logger.info("🏷️  All POS features are zeros - recalculating...")
                needs_pos_calculation = True
        
        if needs_pos_calculation:
            df = calculate_pos_features(df)
        
        # Save the enhanced file
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        new_columns = list(df.columns)
        added_columns = set(new_columns) - set(original_columns)
        
        if added_columns:
            logger.info(f"✅ Enhanced {os.path.basename(csv_path)} - Added: {list(added_columns)}")
        else:
            logger.info(f"✅ Enhanced {os.path.basename(csv_path)} - No new columns needed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error enhancing {os.path.basename(csv_path)}: {e}")
        return False

def enhance_all_prediction_files(predictions_dir=None, reviewed_dir=None):
    """Enhance all prediction CSV files with missing features"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    if predictions_dir is None:
        predictions_dir = os.path.join(base_dir, 'predictions')
    if reviewed_dir is None:
        reviewed_dir = os.path.join(base_dir, 'reviewed')
    
    logger.info("🔧 ENHANCING PREDICTION FILES WITH MISSING FEATURES")
    logger.info("=" * 60)
    
    # Find all prediction CSV files
    prediction_files = []
    
    # Check predictions directory
    if os.path.exists(predictions_dir):
        prediction_files.extend(glob.glob(os.path.join(predictions_dir, "*_predictions.csv")))
    
    # Check reviewed directory
    if os.path.exists(reviewed_dir):
        prediction_files.extend(glob.glob(os.path.join(reviewed_dir, "*_predictions.csv")))
    
    if not prediction_files:
        logger.warning("⚠️  No prediction files found to enhance")
        return
    
    logger.info(f"📄 Found {len(prediction_files)} prediction files to enhance")
    
    success_count = 0
    error_count = 0
    
    for csv_file in prediction_files:
        if enhance_prediction_file(csv_file):
            success_count += 1
        else:
            error_count += 1
    
    logger.info("=" * 60)
    logger.info(f"✅ Enhancement complete!")
    logger.info(f"   ✅ Successfully enhanced: {success_count} files")
    if error_count > 0:
        logger.info(f"   ❌ Errors: {error_count} files")
    
    return success_count, error_count

def main():
    """Main function"""
    print("🔧 ENHANCED FEATURE CALCULATOR FOR RETRAINING")
    print("=" * 60)
    print("🎯 Features:")
    print("   ✅ Calculate distance_to_previous_heading")
    print("   ✅ Calculate POS features (if spaCy available)")
    print("   ✅ Process all prediction CSV files")
    print("   ✅ Handle missing/partial features")
    print("   ✅ Prepare files for retraining")
    print()
    
    try:
        enhance_all_prediction_files()
        
    except KeyboardInterrupt:
        print("\n⚠️  Enhancement cancelled by user")
    except Exception as e:
        logger.error(f"❌ Error in main: {e}")
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
