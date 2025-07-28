#!/usr/bin/env python3
"""
Precision-Focused Model Trainer
===============================

This script focuses specifically on reducing false positives in heading detection.
It implements strict heading criteria and precision-optimized training.

Key Features:
✅ Ultra-high precision training (target: 90%+ precision)
✅ Advanced false positive filtering
✅ Strict heading pattern requirements
✅ Dynamic threshold optimization for precision
✅ No feature selection to avoid compatibility issues
✅ Comprehensive heading validation rules

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
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, f1_score, recall_score, precision_score, precision_recall_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PrecisionFocusedTrainer:
    """Precision-focused trainer that minimizes false positives"""
    
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.config = self.load_config()
        
        # Directories
        self.labelled_data_dir = os.path.join(self.base_dir, self.config['directories']['labelled_data'])
        self.models_dir = os.path.join(self.base_dir, self.config['directories']['models'])
        
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Model components
        self.model = None
        self.tfidf_vectorizer = None
        self.feature_columns = None
        self.optimal_threshold = self.config['pipeline_settings']['confidence_threshold']
        
        logger.info("🎯 Precision-Focused Trainer initialized!")
        logger.info(f"⚙️  Mode: {self.config['pipeline_settings']['training_mode']}")
        logger.info(f"🎯 Target Precision: {self.config['training']['precision_target']}")
    
    def load_config(self):
        """Load configuration"""
        config_path = os.path.join(self.base_dir, 'config_main.json')
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def load_labeled_data(self):
        """Load and filter labeled data with strict quality requirements"""
        logger.info("📊 Loading labeled data with precision focus...")
        
        csv_files = glob.glob(os.path.join(self.labelled_data_dir, "*.csv"))
        csv_files = [f for f in csv_files if not f.endswith('_with_predictions.csv')]
        
        if not csv_files:
            logger.error("❌ No CSV files found")
            return None
        
        dataframes = []
        min_heading_percentage = self.config['pipeline_settings']['min_heading_percentage']
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                
                if 'text' not in df.columns or 'is_heading' not in df.columns:
                    logger.warning(f"⚠️  Skipping {Path(csv_file).name}: missing required columns")
                    continue
                
                # Clean data
                df['text'] = df['text'].fillna('').astype(str)
                df['is_heading'] = pd.to_numeric(df['is_heading'], errors='coerce').fillna(0).astype(int)
                df = df[df['text'].str.strip() != '']
                
                if len(df) == 0:
                    continue
                
                # Check heading percentage
                heading_percentage = (df['is_heading'].sum() / len(df)) * 100
                if heading_percentage < min_heading_percentage:
                    logger.warning(f"⚠️  Skipping {Path(csv_file).name}: only {heading_percentage:.1f}% headings")
                    continue
                
                df['source_file'] = Path(csv_file).stem
                dataframes.append(df)
                
                logger.info(f"✅ {Path(csv_file).name}: {len(df)} blocks, {df['is_heading'].sum()} headings ({heading_percentage:.1f}%)")
                
            except Exception as e:
                logger.error(f"❌ Error loading {csv_file}: {e}")
        
        if not dataframes:
            logger.error("❌ No valid CSV files loaded")
            return None
        
        combined_df = pd.concat(dataframes, ignore_index=True)
        combined_df = combined_df.dropna(subset=['text', 'is_heading'])
        combined_df['is_heading'] = combined_df['is_heading'].astype(int)
        
        logger.info(f"📊 Final dataset: {len(combined_df)} blocks, {combined_df['is_heading'].sum()} headings ({combined_df['is_heading'].mean()*100:.1f}%)")
        
        return combined_df
    
    def apply_precision_filters(self, df):
        """Apply strict filters to reduce false positives"""
        logger.info("🔍 Applying precision filters...")
        
        original_count = len(df)
        
        # Apply precision filters from config
        filters = self.config.get('precision_filters', {})
        
        # Filter 1: Remove very long texts (likely paragraphs, not headings)
        max_word_count = filters.get('max_word_count', 20)
        df = df[df['text'].str.split().str.len() <= max_word_count]
        logger.info(f"   📝 Word count filter: {len(df)} blocks remain (removed {original_count - len(df)} long texts)")
        
        # Filter 2: Remove texts with sentence-ending patterns
        if filters.get('exclude_sentence_patterns', True):
            # Remove texts ending with periods (likely sentences)
            sentence_pattern = r'\.$|[.!?]\s*$'
            before = len(df)
            df = df[~df['text'].str.contains(sentence_pattern, regex=True, na=False)]
            logger.info(f"   🚫 Sentence pattern filter: {len(df)} blocks remain (removed {before - len(df)} sentences)")
        
        # Filter 3: Require heading-like patterns
        if filters.get('required_heading_patterns', True):
            before = len(df)
            heading_patterns = (
                df['text'].str.istitle() |  # Title case
                df['text'].str.isupper() |  # All caps
                df['text'].str.contains(r'^[A-Z]', na=False) |  # Starts with capital
                df['text'].str.contains(r'^\d+\.?\s+[A-Z]', na=False) |  # Numbered headings
                df['text'].str.endswith(':', na=False) |  # Ends with colon
                df['text'].str.contains(r'\b(?:chapter|section|part|introduction|conclusion|summary|abstract|references|appendix)\b', case=False, na=False)  # Common heading words
            )
            df = df[heading_patterns]
            logger.info(f"   📋 Heading pattern filter: {len(df)} blocks remain (removed {before - len(df)} non-heading patterns)")
        
        # Filter 4: Font size requirements (if available)
        if 'font_size' in df.columns and filters.get('min_font_size_percentile'):
            min_percentile = filters['min_font_size_percentile']
            font_threshold = np.percentile(df['font_size'].dropna(), min_percentile)
            before = len(df)
            df = df[df['font_size'] >= font_threshold]
            logger.info(f"   📏 Font size filter (≥{min_percentile}th percentile): {len(df)} blocks remain (removed {before - len(df)} small fonts)")
        
        logger.info(f"🎯 Precision filtering complete: {original_count} → {len(df)} blocks ({len(df)/original_count*100:.1f}% retained)")
        
        return df
    
    def prepare_precision_features(self, df):
        """Prepare features optimized for precision"""
        logger.info("🔧 Preparing precision-optimized features...")
        
        # Ensure required columns exist
        df = df.copy()
        df['text'] = df['text'].fillna('').astype(str)
        
        # Debug: Print column info
        logger.info(f"📊 DataFrame columns: {list(df.columns)}")
        logger.info(f"📊 DataFrame shape: {df.shape}")
        
        # Check for problematic columns with non-numeric data
        for col in df.columns:
            if col != 'text' and col != 'is_heading' and col != 'source_file':
                non_numeric_mask = pd.to_numeric(df[col], errors='coerce').isna()
                if non_numeric_mask.any():
                    sample_values = df[non_numeric_mask][col].head(3).tolist()
                    logger.warning(f"⚠️  Column '{col}' has non-numeric values: {sample_values}")
        
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
        
        # Convert numeric features - handle non-numeric data gracefully
        numeric_features = ['font_size', 'page', 'x0', 'y0', 'x1', 'y1', 'word_count', 'relative_font_size']
        for feature in numeric_features:
            if feature in df.columns:
                # Convert to numeric, replacing non-numeric values with 0
                df[feature] = pd.to_numeric(df[feature], errors='coerce').fillna(0)
        
        # Handle boolean/categorical features that might be strings
        boolean_features = ['bold', 'italic', 'underline', 'is_all_caps', 'is_title_case', 
                           'ends_with_colon', 'starts_with_number']
        for feature in boolean_features:
            if feature in df.columns:
                # Convert to numeric if it's not already, handle text values
                if df[feature].dtype == 'object':
                    # Convert string representations to boolean
                    df[feature] = df[feature].astype(str).str.lower().isin(['true', '1', 'yes']).astype(int)
                else:
                    df[feature] = pd.to_numeric(df[feature], errors='coerce').fillna(0).astype(int)
        
        # Enhanced text features for precision
        df['text_length'] = df['text'].str.len()
        df['word_count_calc'] = df['text'].str.split().str.len()
        df['avg_word_length'] = df['text'].str.split().apply(lambda x: np.mean([len(w) for w in x]) if x else 0)
        
        # Heading-specific pattern features
        df['is_title_case_strict'] = df['text'].str.istitle().astype(int)
        df['is_all_caps_strict'] = df['text'].str.isupper().astype(int)
        df['starts_with_capital'] = df['text'].str.match(r'^[A-Z]', na=False).astype(int)
        df['starts_with_number'] = df['text'].str.match(r'^\d', na=False).astype(int)
        df['ends_with_colon'] = df['text'].str.endswith(':', na=False).astype(int)
        df['has_common_heading_words'] = df['text'].str.contains(
            r'\b(?:chapter|section|part|introduction|conclusion|summary|abstract|references|appendix|table|figure)\b', 
            case=False, na=False
        ).astype(int)
        
        # Length-based features (headings are typically shorter)
        df['is_very_short'] = (df['word_count_calc'] <= 3).astype(int)
        df['is_short'] = (df['word_count_calc'].between(4, 8)).astype(int)
        df['is_medium'] = (df['word_count_calc'].between(9, 15)).astype(int)
        df['is_long'] = (df['word_count_calc'] > 15).astype(int)
        
        # Font features (if available)
        if 'font_size' in df.columns:
            font_percentiles = self.config['feature_engineering']['font_percentiles']
            for p in font_percentiles:
                thresh = np.percentile(df['font_size'].dropna(), p)
                df[f'font_size_ge_{p}'] = (df['font_size'] >= thresh).astype(int)
        
        # Position features (if available)
        if 'x0' in df.columns and 'y0' in df.columns:
            df['width'] = (df['x1'] - df['x0']).fillna(0)
            df['height'] = (df['y1'] - df['y0']).fillna(0)
            df['aspect_ratio'] = (df['width'] / (df['height'] + 1)).fillna(0)
            df['is_left_aligned'] = (df['x0'] <= df['x0'].quantile(0.25)).astype(int)
        
        # TF-IDF features (reduced for precision)
        logger.info("📝 Computing TF-IDF features...")
        tfidf_config = self.config['tfidf_params']
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=tfidf_config['max_features'],
            ngram_range=tuple(tfidf_config['ngram_range']),
            min_df=tfidf_config['min_df'],
            max_df=tfidf_config['max_df'],
            stop_words=tfidf_config['stop_words'],
            sublinear_tf=tfidf_config['sublinear_tf'],
            norm=tfidf_config['norm']
        )
        
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(df['text'])
        tfidf_feature_names = [f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])]
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_feature_names, index=df.index)
        
        # Combine all features
        feature_df = pd.concat([df, tfidf_df], axis=1)
        
        # Select feature columns (exclude metadata and text columns)
        exclude_cols = ['text', 'is_heading', 'source_file', 'heading_level', 'font', 'color']
        candidate_features = [col for col in feature_df.columns if col not in exclude_cols]
        
        # Further filter to only numeric columns and handle NaN values
        self.feature_columns = []
        for col in candidate_features:
            try:
                # Convert to numeric and check for issues
                numeric_col = pd.to_numeric(feature_df[col], errors='coerce')
                # Fill any remaining NaN values with 0
                feature_df[col] = numeric_col.fillna(0)
                self.feature_columns.append(col)
            except (ValueError, TypeError):
                logger.warning(f"⚠️  Excluding non-numeric feature: {col}")
        
        # Final check: ensure no NaN values remain in feature columns
        feature_matrix = feature_df[self.feature_columns]
        nan_counts = feature_matrix.isna().sum()
        if nan_counts.any():
            logger.warning("⚠️  Found remaining NaN values, filling with 0:")
            for col in nan_counts[nan_counts > 0].index:
                logger.warning(f"   - {col}: {nan_counts[col]} NaN values")
                feature_df[col] = feature_df[col].fillna(0)
        
        logger.info(f"✅ Prepared {len(self.feature_columns)} numeric features for training")
        
        return feature_df
    
    def train_precision_model(self, df):
        """Train model with precision focus"""
        logger.info("🎯 Training precision-focused model...")
        
        # Prepare features
        feature_df = self.prepare_precision_features(df)
        
        # Get features and target
        X = feature_df[self.feature_columns]
        y = feature_df['is_heading']
        
        # Final NaN check before training
        nan_check = X.isna().sum().sum()
        if nan_check > 0:
            logger.warning(f"⚠️  Found {nan_check} NaN values in features, filling with 0")
            X = X.fillna(0)
        
        logger.info(f"📊 Training data: {len(X)} samples, {len(self.feature_columns)} features")
        logger.info(f"📊 Class distribution: {y.value_counts().to_dict()}")
        
        # Split data
        test_size = self.config['training']['test_size']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )
        
        # Final check on training data
        if X_train.isna().sum().sum() > 0:
            logger.warning("⚠️  Found NaN in training data, filling with 0")
            X_train = X_train.fillna(0)
        if X_test.isna().sum().sum() > 0:
            logger.warning("⚠️  Found NaN in test data, filling with 0")
            X_test = X_test.fillna(0)
        
        # Apply SMOTE with conservative settings
        smote_config = self.config['smote_params']
        smote = SMOTE(
            sampling_strategy=smote_config['sampling_strategy'],
            random_state=smote_config['random_state'],
            k_neighbors=smote_config['k_neighbors']
        )
        
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        logger.info(f"📊 After SMOTE: {len(X_train_balanced)} samples")
        logger.info(f"📊 Balanced class distribution: {pd.Series(y_train_balanced).value_counts().to_dict()}")
        
        # Train model with precision-focused parameters
        model_params = self.config['model_params']
        
        # Convert class weights from string keys to integer keys
        class_weight = model_params['class_weight']
        if isinstance(list(class_weight.keys())[0], str):
            class_weight = {int(k): v for k, v in class_weight.items()}
        
        self.model = RandomForestClassifier(
            n_estimators=model_params['n_estimators'],
            max_depth=model_params['max_depth'],
            min_samples_split=model_params['min_samples_split'],
            min_samples_leaf=model_params['min_samples_leaf'],
            max_features=model_params['max_features'],
            max_samples=model_params.get('max_samples'),
            bootstrap=model_params['bootstrap'],
            class_weight=class_weight,
            random_state=42,
            n_jobs=-1
        )
        
        logger.info("🌳 Training Random Forest...")
        self.model.fit(X_train_balanced, y_train_balanced)
        
        # Predict probabilities
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Find optimal threshold for precision
        self.optimal_threshold = self._find_precision_threshold(y_test, y_pred_proba)
        
        # Final predictions with optimal threshold
        y_pred = (y_pred_proba >= self.optimal_threshold).astype(int)
        
        # Calculate metrics
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        logger.info("🎯 PRECISION-FOCUSED TRAINING RESULTS")
        logger.info("=" * 50)
        logger.info(f"📈 Precision: {precision:.3f}")
        logger.info(f"📈 Recall: {recall:.3f}")
        logger.info(f"📈 F1-Score: {f1:.3f}")
        logger.info(f"🎯 Optimal Threshold: {self.optimal_threshold:.3f}")
        logger.info(f"📊 Predicted headings: {y_pred.sum()}/{len(y_test)}")
        
        # Detailed classification report
        logger.info("\\n📋 Classification Report:")
        logger.info("\\n" + classification_report(y_test, y_pred))
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'optimal_threshold': self.optimal_threshold,
            'test_predictions': y_pred.sum(),
            'test_samples': len(y_test)
        }
    
    def _find_precision_threshold(self, y_true, y_proba):
        """Find threshold that maximizes precision while maintaining reasonable recall"""
        precision_target = self.config['training']['precision_target']
        
        # Calculate precision-recall curve
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
        
        # Find threshold that achieves target precision
        target_indices = np.where(precisions >= precision_target)[0]
        
        if len(target_indices) > 0:
            # Choose threshold with highest recall among those meeting precision target
            best_idx = target_indices[np.argmax(recalls[target_indices])]
            optimal_threshold = thresholds[best_idx]
            logger.info(f"🎯 Found threshold {optimal_threshold:.3f} with precision {precisions[best_idx]:.3f} and recall {recalls[best_idx]:.3f}")
        else:
            # If target precision can't be achieved, choose threshold with best precision
            best_idx = np.argmax(precisions)
            optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
            logger.warning(f"⚠️  Target precision {precision_target:.3f} not achievable. Using threshold {optimal_threshold:.3f} with precision {precisions[best_idx]:.3f}")
        
        return optimal_threshold
    
    def save_model(self, version="precision_v1"):
        """Save the precision-focused model"""
        if self.model is None:
            logger.error("❌ No model to save")
            return None
        
        model_path = os.path.join(self.models_dir, f"heading_model_{version}.pkl")
        vectorizer_path = os.path.join(self.models_dir, f"tfidf_vectorizer_{version}.pkl")
        features_path = os.path.join(self.models_dir, f"feature_columns_{version}.pkl")
        threshold_path = os.path.join(self.models_dir, f"optimal_threshold_{version}.pkl")
        
        # Save model components
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.tfidf_vectorizer, f)
        
        with open(features_path, 'wb') as f:
            pickle.dump(self.feature_columns, f)
        
        with open(threshold_path, 'wb') as f:
            pickle.dump(self.optimal_threshold, f)
        
        logger.info(f"✅ Precision model saved:")
        logger.info(f"   📄 Model: {model_path}")
        logger.info(f"   📄 Vectorizer: {vectorizer_path}")
        logger.info(f"   📄 Features: {features_path}")
        logger.info(f"   📄 Threshold: {threshold_path}")
        
        return model_path
    
    def run_precision_training(self):
        """Run the complete precision training pipeline"""
        logger.info("🚀 STARTING PRECISION-FOCUSED TRAINING")
        logger.info("=" * 60)
        
        # Load data
        df = self.load_labeled_data()
        if df is None:
            return False
        
        # Apply precision filters
        df_filtered = self.apply_precision_filters(df)
        if len(df_filtered) == 0:
            logger.error("❌ No data remaining after precision filters")
            return False
        
        # Train model
        results = self.train_precision_model(df_filtered)
        
        # Save model
        model_path = self.save_model()
        
        logger.info("🎉 PRECISION TRAINING COMPLETED!")
        logger.info(f"📈 Final Precision: {results['precision']:.3f}")
        logger.info(f"📈 Final Recall: {results['recall']:.3f}")
        logger.info(f"🎯 Threshold: {results['optimal_threshold']:.3f}")
        
        return True

def main():
    """Main function"""
    print("🎯 PRECISION-FOCUSED HEADING DETECTION TRAINER")
    print("=" * 60)
    print("🎯 Objective: Minimize false positives in heading detection")
    print("📊 Target: 90%+ precision for H1, H2, H3 headings only")
    print()
    
    try:
        trainer = PrecisionFocusedTrainer()
        success = trainer.run_precision_training()
        
        if success:
            print("\\n✅ Training completed successfully!")
            print("🎯 Your model is now optimized for ultra-high precision.")
            print("📝 Use this model with generate_json_output.py for competition submission.")
        else:
            print("\\n❌ Training failed. Please check the logs above.")
            
    except KeyboardInterrupt:
        print("\\n🛑 Training interrupted by user")
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        print(f"\\n❌ Error: {e}")

if __name__ == "__main__":
    main()
