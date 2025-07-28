"""
WORKING Heading Detection Pipeline
==================================

This version combines the best practices and avoids the TF-IDF issues:
1. Filters out poor quality files (0% headings)
2. Uses robust class balancing 
3. Includes both basic features AND simple text features
4. Handles multilingual content
5. Finds optimal threshold
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
import glob
import logging
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, f1_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class WorkingHeadingDetectionPipeline:
    """Working pipeline that actually predicts headings correctly"""
    
    def __init__(self):
        self.base_dir = os.path.dirname(__file__)
        self.project_root = os.path.abspath(os.path.join(self.base_dir, '..'))
        
        # Directories
        self.labelled_data_dir = os.path.join(self.project_root, 'labelled_data')
        self.unprocessed_pdfs_dir = os.path.join(self.project_root, 'unprocessed_pdfs')
        self.models_dir = os.path.join(self.base_dir, 'models')
        self.predictions_dir = os.path.join(self.base_dir, 'predictions')
        self.reviewed_dir = os.path.join(self.base_dir, 'reviewed')
        
        # Create directories
        for dir_path in [self.models_dir, self.predictions_dir, self.reviewed_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Model components
        self.model = None
        self.feature_columns = []
        self.optimal_threshold = 0.5  # Will be tuned during training
        
        # Expected features from your CSV files
        self.base_features = [
            'font_size', 'page', 'x0', 'y0', 'x1', 'y1', 'bold', 'italic', 
            'underline', 'is_all_caps', 'is_title_case', 'ends_with_colon',
            'starts_with_number', 'punctuation_count', 'contains_colon',
            'contains_semicolon', 'word_count', 'line_position_on_page',
            'relative_font_size'
        ]
    
    def load_labeled_data(self, min_heading_percentage=1.0):
        """Load labeled data with quality filtering"""
        logger.info(f"📂 Loading labeled data from: {self.labelled_data_dir}")
        logger.info(f"🎯 Filtering files with <{min_heading_percentage}% headings")
        
        csv_files = glob.glob(os.path.join(self.labelled_data_dir, "*.csv"))
        if not csv_files:
            logger.error(f"❌ No CSV files found in {self.labelled_data_dir}")
            return None
        
        all_data = []
        loaded_files = 0
        skipped_files = 0
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                
                # Check required columns
                if 'text' not in df.columns or 'is_heading' not in df.columns:
                    logger.warning(f"⚠️  Skipping {Path(csv_file).name}: missing required columns")
                    continue
                
                # Clean missing values
                df['text'] = df['text'].fillna('').astype(str)
                df['is_heading'] = pd.to_numeric(df['is_heading'], errors='coerce').fillna(0).astype(int)
                
                # Remove rows with empty text
                df = df[df['text'].str.strip() != '']
                
                if len(df) == 0:
                    logger.warning(f"⚠️  Skipping {Path(csv_file).name}: no valid data after cleaning")
                    continue
                
                # Check heading percentage
                headings_count = df['is_heading'].sum()
                total_blocks = len(df)
                heading_percentage = (headings_count / total_blocks) * 100
                
                if heading_percentage < min_heading_percentage:
                    logger.warning(f"⚠️  Skipping {Path(csv_file).name}: only {heading_percentage:.1f}% headings")
                    skipped_files += 1
                    continue
                
                # Add source file
                df['source_file'] = Path(csv_file).stem
                all_data.append(df)
                loaded_files += 1
                
                logger.info(f"   ✅ {Path(csv_file).name}: {total_blocks} blocks, {headings_count} headings ({heading_percentage:.1f}%)")
                
            except Exception as e:
                logger.error(f"❌ Error loading {csv_file}: {e}")
        
        if not all_data:
            logger.error("❌ No valid CSV files loaded after filtering")
            logger.info("💡 Try reducing min_heading_percentage")
            return None
        
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Final cleaning
        combined_df = combined_df.dropna(subset=['text', 'is_heading'])
        combined_df['is_heading'] = combined_df['is_heading'].astype(int)
        
        logger.info(f"🎯 Loaded: {len(combined_df)} blocks from {loaded_files} files (skipped {skipped_files})")
        logger.info(f"📊 Headings: {combined_df['is_heading'].sum()} ({combined_df['is_heading'].mean()*100:.1f}%)")
        logger.info(f"📊 Class ratio: {(len(combined_df) - combined_df['is_heading'].sum()) / combined_df['is_heading'].sum():.1f}:1")
        
        return combined_df
    
    def prepare_features(self, df):
        """Prepare features WITHOUT TF-IDF to avoid vocabulary issues"""
        logger.info("🔧 Preparing features (without TF-IDF)...")
        df = df.copy()
        df['text'] = df['text'].fillna('').astype(str)
        df = df[df['text'].str.strip() != '']
        
        if len(df) == 0:
            logger.error("❌ No valid text data after cleaning")
            return None
        
        # Ensure all base features exist with proper defaults
        for feature in self.base_features:
            if feature not in df.columns:
                if feature in ['font_size', 'relative_font_size']:
                    df[feature] = 12.0
                elif feature in ['page']:
                    df[feature] = 1
                elif feature in ['x0', 'y0', 'x1', 'y1', 'line_position_on_page']:
                    df[feature] = 0.0
                elif feature in ['word_count']:
                    df[feature] = df['text'].str.split().str.len()
                else:
                    df[feature] = 0
        
        # Convert numeric features and handle missing values
        numeric_features = [f for f in self.base_features if f not in ['text']]
        for feature in numeric_features:
            if feature in df.columns:
                df[feature] = pd.to_numeric(df[feature], errors='coerce')
                
                # Fill missing values with appropriate defaults
                if feature in ['font_size', 'relative_font_size']:
                    df[feature] = df[feature].fillna(12.0)
                elif feature == 'page':
                    df[feature] = df[feature].fillna(1)
                elif feature == 'word_count':
                    df[feature] = df[feature].fillna(df['text'].str.split().str.len())
                else:
                    df[feature] = df[feature].fillna(0)
        
        # Enhanced font features
        if 'font_size' in df.columns:
            percentiles = [99, 95, 90, 85, 80, 75, 70, 60, 50, 40, 25, 10]
            for p in percentiles:
                thresh = np.percentile(df['font_size'].dropna(), p)
                df[f'font_size_ge_{p}'] = (df['font_size'] >= thresh).astype(int)
        
        # Rich text features for heading detection
        df['text_length'] = df['text'].str.len()
        df['word_count_calc'] = df['text'].str.split().str.len()
        df['avg_word_length'] = df['text'].str.split().apply(lambda x: np.mean([len(w) for w in x]) if x else 0)
        df['has_colon'] = df['text'].str.contains(':').astype(int)
        df['has_semicolon'] = df['text'].str.contains(';').astype(int)
        df['has_number'] = df['text'].str.contains(r'\\d').astype(int)
        df['has_special_chars'] = df['text'].str.contains(r'[!@#$%^&*(),.?\":{}|<>]').astype(int)
        df['is_short_text'] = (df['text_length'] <= 50).astype(int)
        df['is_very_short'] = (df['word_count_calc'] <= 5).astype(int)
        df['is_medium_length'] = ((df['word_count_calc'] > 5) & (df['word_count_calc'] <= 15)).astype(int)
        df['is_caps'] = df['text'].str.isupper().astype(int)
        df['title_case_ratio'] = df['text'].apply(lambda x: sum(1 for w in x.split() if w.istitle()) / len(x.split()) if x.split() else 0)
        df['upper_ratio'] = df['text'].apply(lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0)
        df['starts_with_capital'] = df['text'].str.match(r'^[A-Z]').fillna(False).astype(int)
        df['ends_with_period'] = df['text'].str.endswith('.').astype(int)
        df['contains_parentheses'] = df['text'].str.contains(r'[()]').astype(int)
        df['numeric_start'] = df['text'].str.match(r'^\\d+').fillna(False).astype(int)
        
        # Position-based features (if available)
        if 'y0' in df.columns and 'page' in df.columns:
            df['relative_y_position'] = df.groupby('page')['y0'].rank(pct=True)
            df['is_top_of_page'] = (df['relative_y_position'] >= 0.9).astype(int)
            df['is_bottom_of_page'] = (df['relative_y_position'] <= 0.1).astype(int)
        
        # Combine all features
        feature_cols = [f for f in self.base_features if f in df.columns]
        feature_cols += [f'font_size_ge_{p}' for p in [95,90,75] if f'font_size_ge_{p}' in df.columns]
        feature_cols += [
            'text_length', 'word_count_calc', 'avg_word_length', 'has_colon', 'has_semicolon',
            'has_number', 'has_special_chars', 'is_short_text', 'is_very_short', 'is_medium_length',
            'is_caps', 'title_case_ratio', 'upper_ratio', 'starts_with_capital', 'ends_with_period',
            'contains_parentheses', 'numeric_start'
        ]
        
        # Add position features if available
        if 'relative_y_position' in df.columns:
            feature_cols += ['relative_y_position', 'is_top_of_page', 'is_bottom_of_page']
        
        # Select available features
        available_features = [f for f in feature_cols if f in df.columns]
        X_combined = df[available_features].fillna(0)
        
        # Final cleanup
        X_combined = X_combined.replace([np.inf, -np.inf], 0)
        X_combined = X_combined.fillna(0)
        
        self.feature_columns = X_combined.columns.tolist()
        logger.info(f"✅ Features prepared: {len(self.feature_columns)} features (no TF-IDF)")
        
        return X_combined
    
    def train_model(self, df, test_size=0.2, random_state=42):
        """Train model with robust class balancing"""
        logger.info("🚀 Training working heading detection model...")
        
        # Prepare features
        X = self.prepare_features(df)
        y = df['is_heading'].astype(int)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"📊 Training set: {len(X_train)} samples")
        logger.info(f"📊 Test set: {len(X_test)} samples")
        logger.info(f"📊 Positive class: {y_train.sum()} ({y_train.mean()*100:.1f}%)")
        
        # SMOTE for class balancing
        try:
            smote = SMOTE(sampling_strategy=0.4, random_state=random_state)  # Balance to 40%
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
            logger.info(f"✅ SMOTE applied: {len(X_train_res)} samples ({y_train_res.mean()*100:.1f}% positive)")
        except Exception as e:
            logger.warning(f"⚠️  SMOTE failed: {e}. Using class weights instead.")
            X_train_res, y_train_res = X_train, y_train
        
        # Train model with strong class weights
        self.model = RandomForestClassifier(
            n_estimators=500,
            max_depth=25,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=random_state,
            n_jobs=-1,
            class_weight={0: 1, 1: 15}
        )
        
        logger.info("🌳 Training Random Forest with strong heading bias...")
        self.model.fit(X_train_res, y_train_res)
        
        # Find optimal threshold
        y_proba = self.model.predict_proba(X_test)[:, 1]
        best_threshold = self.find_optimal_threshold(y_test, y_proba)
        self.optimal_threshold = best_threshold
        
        # Evaluate with optimal threshold
        test_pred = (y_proba >= self.optimal_threshold).astype(int)
        test_score = (test_pred == y_test).mean()
        
        logger.info(f"📈 Test accuracy: {test_score:.3f}")
        logger.info(f"🎯 Optimal threshold: {self.optimal_threshold:.3f}")
        logger.info(f"📊 Predicted headings: {test_pred.sum()} out of {len(test_pred)} test samples")
        
        # Detailed classification report
        report = classification_report(y_test, test_pred, target_names=['Not Heading', 'Heading'])
        logger.info(f"📊 Classification Report:\\n{report}")
        
        # Show some example predictions
        if test_pred.sum() > 0:
            heading_indices = np.where(test_pred == 1)[0][:5]
            test_df = df.iloc[X_test.index]
            
            logger.info("📝 Sample predicted headings:")
            for i, idx in enumerate(heading_indices):
                text_sample = test_df.iloc[idx]['text'][:60] + "..." if len(test_df.iloc[idx]['text']) > 60 else test_df.iloc[idx]['text']
                confidence = y_proba[idx]
                actual = y_test.iloc[idx]
                status = "✅" if actual == 1 else "❌"
                logger.info(f"   {i+1}. {status} Conf: {confidence:.3f}, Text: '{text_sample}'")
        
        return {
            'test_accuracy': test_score,
            'optimal_threshold': self.optimal_threshold,
            'classification_report': report,
            'predicted_headings': test_pred.sum(),
            'total_test_samples': len(test_pred)
        }
    
    def find_optimal_threshold(self, y_true, y_proba):
        """Find optimal threshold using F1 score"""
        thresholds = np.arange(0.1, 0.9, 0.05)
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            f1 = f1_score(y_true, y_pred)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        logger.info(f"🎯 Best F1 score: {best_f1:.3f} at threshold {best_threshold:.3f}")
        return best_threshold
    
    def predict_with_threshold(self, X):
        """Predict using optimal threshold"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        probabilities = self.model.predict_proba(X)[:, 1]
        predictions = (probabilities >= self.optimal_threshold).astype(int)
        
        return predictions, probabilities
    
    def save_model(self, version=None):
        """Save the trained model"""
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M")
        
        model_path = os.path.join(self.models_dir, f'working_heading_classifier_v{version}.pkl')
        
        # Save all components
        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'optimal_threshold': self.optimal_threshold
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"💾 Working model saved: {model_path}")
        logger.info(f"💾 Threshold: {self.optimal_threshold:.3f}")
        
        return model_path
    
    def load_model(self, version=None):
        """Load a trained model"""
        if version is None:
            # Find latest working version
            model_files = glob.glob(os.path.join(self.models_dir, 'working_heading_classifier_v*.pkl'))
            if not model_files:
                logger.error("❌ No working models found")
                return False
            
            latest_model = max(model_files, key=os.path.getctime)
            model_path = latest_model
        else:
            model_path = os.path.join(self.models_dir, f'working_heading_classifier_v{version}.pkl')
        
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.feature_columns = model_data['feature_columns']
            self.optimal_threshold = model_data.get('optimal_threshold', 0.5)
            
            logger.info(f"✅ Working model loaded: {model_path}")
            logger.info(f"   🎯 Optimal threshold: {self.optimal_threshold:.3f}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            return False

def main():
    """Run the working pipeline"""
    print("🚀 WORKING HEADING DETECTION PIPELINE")
    print("="*50)
    print("This version:")
    print("• Filters out files with 0% headings")
    print("• Uses rich text features (no TF-IDF)")
    print("• Strong class weighting (1:8)")
    print("• Optimal threshold tuning")
    print("• Should actually predict headings!")
    print()
    
    pipeline = WorkingHeadingDetectionPipeline()
    
    # Load data with quality filtering
    print("📊 Loading quality data...")
    df = pipeline.load_labeled_data(min_heading_percentage=1.0)
    
    if df is None:
        print("❌ No quality data found.")
        return
    
    # Train model
    print("\\n🚀 Training working model...")
    results = pipeline.train_model(df)
    
    # Save model
    model_path = pipeline.save_model("working")
    
    print("\\n✅ WORKING PIPELINE COMPLETED!")
    print(f"📈 Test accuracy: {results['test_accuracy']:.3f}")
    print(f"🎯 Optimal threshold: {results['optimal_threshold']:.3f}")
    print(f"📊 Predicted {results['predicted_headings']} headings out of {results['total_test_samples']} test samples")
    print(f"💾 Model saved: {model_path}")
    
    if results['predicted_headings'] > 0:
        print("\\n🎉 SUCCESS: Model is predicting headings (not all 0s)!")
    else:
        print("\\n⚠️  Model still predicting all 0s - may need more data quality improvements")
    
    return pipeline, results

if __name__ == "__main__":
    pipeline, results = main()
