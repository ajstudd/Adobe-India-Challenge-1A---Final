#!/usr/bin/env python3
"""
Comprehensive Accuracy Testing and Logging System
================================================

This module provides robust accuracy testing with detailed logging,
cross-validation, and comprehensive metrics analysis.
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
import json
import logging
from datetime import datetime
from pathlib import Path
import glob

# Machine Learning imports
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.ensemble import RandomForestClassifier

# Optional visualization imports
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATIONS_AVAILABLE = True
except ImportError:
    VISUALIZATIONS_AVAILABLE = False
    print("⚠️  Visualization libraries not available (matplotlib/seaborn)")
    print("   Installing: pip install matplotlib seaborn")
    print("   Continuing without visualizations...")

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

class AccuracyLogger:
    """Comprehensive accuracy testing and logging system"""
    
    def __init__(self):
        self.base_dir = os.path.dirname(__file__)
        self.accuracy_logs_dir = os.path.join(self.base_dir, 'accuracy_logs')
        self.models_dir = os.path.join(self.base_dir, 'models')
        
        # Create directories
        os.makedirs(self.accuracy_logs_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Setup logging
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.accuracy_logs_dir, f'accuracy_test_{self.timestamp}.log')
        self.json_file = os.path.join(self.accuracy_logs_dir, f'accuracy_test_{self.timestamp}.json')
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Results storage
        self.results = {
            'timestamp': self.timestamp,
            'dataset_info': {},
            'model_info': {},
            'training_results': {},
            'test_results': {},
            'cross_validation': {},
            'detailed_metrics': {},
            'feature_importance': {},
            'threshold_analysis': {},
            'recommendations': []
        }
        
        self.logger.info("🔍 COMPREHENSIVE ACCURACY TESTING INITIALIZED")
        self.logger.info(f"📄 Log file: {self.log_file}")
        self.logger.info(f"📋 JSON report: {self.json_file}")
        if not VISUALIZATIONS_AVAILABLE:
            self.logger.info("⚠️  Running without visualization support")
        self.logger.info("=" * 70)
    
    def analyze_dataset(self):
        """Analyze the training dataset comprehensively"""
        self.logger.info("📊 DATASET ANALYSIS")
        self.logger.info("-" * 50)
        
        # Load all CSV files
        csv_files = glob.glob(os.path.join(self.base_dir, 'labelled_data', '*.csv'))
        csv_files = [f for f in csv_files if not f.endswith('_with_predictions.csv')]
        
        self.logger.info(f"📄 Found {len(csv_files)} CSV files")
        
        total_samples = 0
        total_headings = 0
        file_stats = []
        quality_files = []
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                if 'is_heading' in df.columns:
                    headings = int(df['is_heading'].sum())  # Convert to int
                    percentage = float((headings / len(df)) * 100)  # Convert to float
                    total_samples += len(df)
                    total_headings += headings
                    
                    file_info = {
                        'filename': os.path.basename(csv_file),
                        'samples': int(len(df)),
                        'headings': int(headings),
                        'percentage': float(percentage)
                    }
                    file_stats.append(file_info)
                    
                    # Quality threshold (at least 1% headings)
                    if percentage >= 1.0:
                        quality_files.append(csv_file)
                    
                    self.logger.info(f"   {file_info['filename']:<35} {file_info['samples']:>6} samples, {file_info['headings']:>4} headings ({file_info['percentage']:>5.1f}%)")
            except Exception as e:
                self.logger.error(f"❌ Error loading {csv_file}: {e}")
        
        overall_percentage = float((total_headings / total_samples) * 100)
        
        self.logger.info("-" * 50)
        self.logger.info(f"📊 TOTAL DATASET SUMMARY:")
        self.logger.info(f"   Total CSV files: {len(csv_files)}")
        self.logger.info(f"   Quality files (≥1% headings): {len(quality_files)}")
        self.logger.info(f"   Total samples: {total_samples:,}")
        self.logger.info(f"   Total headings: {total_headings:,}")
        self.logger.info(f"   Overall heading percentage: {overall_percentage:.2f}%")
        
        # Store dataset info
        self.results['dataset_info'] = {
            'total_files': int(len(csv_files)),
            'quality_files': int(len(quality_files)),
            'total_samples': int(total_samples),
            'total_headings': int(total_headings),
            'heading_percentage': float(overall_percentage),
            'file_breakdown': file_stats
        }
        
        # Data quality assessment
        if overall_percentage < 2.0:
            self.logger.warning("⚠️  Very low heading percentage (<2%). This may impact model performance.")
            self.results['recommendations'].append("Consider adding more labeled data with higher heading percentage")
        
        if len(quality_files) < 10:
            self.logger.warning(f"⚠️  Only {len(quality_files)} files have ≥1% headings. More diverse data recommended.")
            self.results['recommendations'].append("Add more diverse documents with clear heading structure")
        
        return quality_files
    
    def load_and_prepare_data(self, quality_files=None):
        """Load and prepare data with quality filtering"""
        self.logger.info("\n🔄 DATA LOADING AND PREPARATION")
        self.logger.info("-" * 50)
        
        if quality_files is None:
            quality_files = self.analyze_dataset()
        
        # Load quality files
        dataframes = []
        for csv_file in quality_files:
            try:
                df = pd.read_csv(csv_file)
                if 'is_heading' in df.columns:
                    dataframes.append(df)
                    self.logger.info(f"✅ Loaded {os.path.basename(csv_file)}: {len(df)} samples")
            except Exception as e:
                self.logger.error(f"❌ Error loading {csv_file}: {e}")
        
        if not dataframes:
            self.logger.error("❌ No valid data loaded!")
            return None, None
        
        # Combine data
        combined_df = pd.concat(dataframes, ignore_index=True)
        self.logger.info(f"📊 Combined dataset: {len(combined_df)} samples")
        
        # Use master pipeline's enhanced feature preparation
        self.logger.info("🔧 Preparing enhanced features with master pipeline...")
        
        try:
            from master_pipeline import MasterPipeline
            
            # Create pipeline with advanced configuration for full feature set
            pipeline = MasterPipeline('config_advanced_pos.json')
            self.logger.info("✅ Using advanced configuration with POS features")
            
            # Use master pipeline's enhanced feature preparation
            X = pipeline.prepare_features(combined_df)
            y = combined_df['is_heading'].astype(int)
            
            # Log feature breakdown
            pos_feature_count = len([c for c in X.columns if any(pos in c for pos in ['num_', 'ratio', 'pos_', 'noun', 'verb', 'adj'])])
            tfidf_feature_count = len([c for c in X.columns if 'tfidf_' in c])
            
            self.logger.info(f"🎯 Enhanced features breakdown:")
            self.logger.info(f"   🏷️  POS-related features: {pos_feature_count}")
            self.logger.info(f"   📄 TF-IDF features: {tfidf_feature_count}")
            self.logger.info(f"   📊 Total features: {X.shape[1]}")
            
        except Exception as e:
            self.logger.warning(f"⚠️  Failed to use master pipeline features: {e}")
            self.logger.info("🔄 Falling back to simplified feature preparation...")
            
            # Fallback to simplified features
            features = {}
            
            # Basic text features
            features['text_length'] = combined_df['text'].str.len().fillna(0)
            features['word_count'] = combined_df['text'].str.split().str.len().fillna(0)
            features['is_numeric'] = combined_df['text'].str.contains(r'\d', na=False).astype(int)
            features['is_upper'] = combined_df['text'].str.isupper().fillna(False).astype(int)
            features['is_title'] = combined_df['text'].str.istitle().fillna(False).astype(int)
            features['starts_with_number'] = combined_df['text'].str.match(r'^\d', na=False).astype(int)
            features['has_punctuation'] = combined_df['text'].str.contains(r'[.!?]', na=False).astype(int)
            
            # Font size features (handle missing column)
            if 'font_size' in combined_df.columns:
                features['font_size'] = combined_df['font_size'].fillna(12.0)
                
                # Font size percentiles
                font_percentiles = [25, 50, 75, 90, 95]
                for p in font_percentiles:
                    threshold = np.percentile(combined_df['font_size'].dropna(), p)
                    features[f'font_ge_{p}p'] = (combined_df['font_size'] >= threshold).astype(int)
            else:
                self.logger.warning("⚠️  'font_size' column missing, using default values")
                features['font_size'] = 12.0
                for p in [25, 50, 75, 90, 95]:
                    features[f'font_ge_{p}p'] = 0
            
            # Position features (handle missing columns)
            if 'x0' in combined_df.columns and 'y0' in combined_df.columns:
                features['x'] = combined_df['x0'].fillna(0)
                features['y'] = combined_df['y0'].fillna(0)
                # Additional position features from bounding box
                if 'x1' in combined_df.columns and 'y1' in combined_df.columns:
                    features['width'] = combined_df['x1'].fillna(0) - combined_df['x0'].fillna(0)
                    features['height'] = combined_df['y1'].fillna(0) - combined_df['y0'].fillna(0)
                    features['center_x'] = (combined_df['x0'].fillna(0) + combined_df['x1'].fillna(0)) / 2
                    features['center_y'] = (combined_df['y0'].fillna(0) + combined_df['y1'].fillna(0)) / 2
                else:
                    features['width'] = 100
                    features['height'] = 20
                    features['center_x'] = features['x']
                    features['center_y'] = features['y']
            elif 'x' in combined_df.columns and 'y' in combined_df.columns:
                # Fallback to legacy x,y columns if available
                features['x'] = combined_df['x'].fillna(0)
                features['y'] = combined_df['y'].fillna(0)
                features['width'] = 100
                features['height'] = 20
                features['center_x'] = features['x']
                features['center_y'] = features['y']
            else:
                self.logger.warning("⚠️  Position columns (x0, y0, x1, y1 or x, y) missing, using default values")
                features['x'] = 0
                features['y'] = 0
                features['width'] = 100
                features['height'] = 20
                features['center_x'] = 0
                features['center_y'] = 0
            
            # Page features (handle missing column)
            if 'page' in combined_df.columns:
                features['page'] = combined_df['page'].fillna(1)
            elif 'page_num' in combined_df.columns:
                features['page'] = combined_df['page_num'].fillna(1)
            else:
                self.logger.warning("⚠️  Page column missing, using default values")
                features['page'] = 1
            
            X = pd.DataFrame(features)
            y = combined_df['is_heading'].astype(int)
        
        self.logger.info(f"✅ Features prepared: {X.shape[1]} features")
        self.logger.info(f"📊 Target distribution: {np.bincount(y)} (0=non-heading, 1=heading)")
        
        return X, y
    
    def comprehensive_model_testing(self, X, y):
        """Comprehensive model testing with multiple metrics"""
        self.logger.info("\n🧪 COMPREHENSIVE MODEL TESTING")
        self.logger.info("-" * 50)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.logger.info(f"📊 Training set: {len(X_train)} samples ({y_train.sum()} headings)")
        self.logger.info(f"📊 Test set: {len(X_test)} samples ({y_test.sum()} headings)")
        
        # Train model with strong class weighting
        self.logger.info("🌳 Training Random Forest model...")
        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=25,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1,
            class_weight={0: 1, 1: 20}  # Strong bias toward headings
        )
        
        model.fit(X_train, y_train)
        self.logger.info("✅ Model training completed")
        
        # Store model info
        self.results['model_info'] = {
            'model_type': 'RandomForestClassifier',
            'n_estimators': 500,
            'max_depth': 25,
            'class_weight': {0: 1, 1: 20},
            'n_features': int(X.shape[1]),
            'feature_names': list(X.columns)
        }
        
        return model, X_train, X_test, y_train, y_test
    
    def evaluate_model_performance(self, model, X_train, X_test, y_train, y_test):
        """Comprehensive model evaluation"""
        self.logger.info("\n📈 MODEL PERFORMANCE EVALUATION")
        self.logger.info("-" * 50)
        
        # Training predictions
        y_train_pred = model.predict(X_train)
        y_train_proba = model.predict_proba(X_train)[:, 1]
        
        # Test predictions
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate comprehensive metrics
        train_metrics = self._calculate_metrics(y_train, y_train_pred, y_train_proba, "Training")
        test_metrics = self._calculate_metrics(y_test, y_test_pred, y_test_proba, "Test")
        
        # Store results
        self.results['training_results'] = train_metrics
        self.results['test_results'] = test_metrics
        
        # Detailed analysis
        self._detailed_analysis(y_test, y_test_pred, y_test_proba)
        
        return model
    
    def _calculate_metrics(self, y_true, y_pred, y_proba, set_name):
        """Calculate comprehensive metrics"""
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1_score': float(f1_score(y_true, y_pred, zero_division=0)),
            'predicted_headings': int(y_pred.sum()),
            'actual_headings': int(y_true.sum()),
            'total_samples': int(len(y_true))
        }
        
        # ROC AUC if we have both classes
        if len(np.unique(y_true)) > 1:
            metrics['roc_auc'] = float(roc_auc_score(y_true, y_proba))
            metrics['avg_precision'] = float(average_precision_score(y_true, y_proba))
        else:
            metrics['roc_auc'] = None
            metrics['avg_precision'] = None
        
        # Log metrics
        self.logger.info(f"📊 {set_name} Metrics:")
        self.logger.info(f"   Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        self.logger.info(f"   Precision: {metrics['precision']:.4f}")
        self.logger.info(f"   Recall:    {metrics['recall']:.4f}")
        self.logger.info(f"   F1-Score:  {metrics['f1_score']:.4f}")
        if metrics['roc_auc']:
            self.logger.info(f"   ROC AUC:   {metrics['roc_auc']:.4f}")
        self.logger.info(f"   Predicted: {metrics['predicted_headings']}/{metrics['total_samples']} headings")
        
        return metrics
    
    def _detailed_analysis(self, y_test, y_test_pred, y_test_proba):
        """Detailed analysis and visualization"""
        self.logger.info("\n🔍 DETAILED ANALYSIS")
        self.logger.info("-" * 50)
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_test_pred)
        self.logger.info("📊 Confusion Matrix:")
        self.logger.info(f"   True Neg: {cm[0,0]:4d} | False Pos: {cm[0,1]:4d}")
        self.logger.info(f"   False Neg: {cm[1,0]:4d} | True Pos:  {cm[1,1]:4d}")
        
        # Classification Report
        report = classification_report(y_test, y_test_pred, target_names=['Not Heading', 'Heading'], output_dict=True)
        self.logger.info("\n📋 Classification Report:")
        self.logger.info(f"   Not Heading - Precision: {report['Not Heading']['precision']:.3f}, Recall: {report['Not Heading']['recall']:.3f}, F1: {report['Not Heading']['f1-score']:.3f}")
        self.logger.info(f"   Heading     - Precision: {report['Heading']['precision']:.3f}, Recall: {report['Heading']['recall']:.3f}, F1: {report['Heading']['f1-score']:.3f}")
        
        # Store detailed metrics
        self.results['detailed_metrics'] = {
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }
        
        # Save visualizations
        self._save_visualizations(y_test, y_test_pred, y_test_proba, cm)
    
    def _save_visualizations(self, y_test, y_test_pred, y_test_proba, cm):
        """Save visualization plots"""
        if not VISUALIZATIONS_AVAILABLE:
            self.logger.info("📊 Skipping visualizations (matplotlib/seaborn not available)")
            return
            
        try:
            plt.style.use('default')
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Model Performance Analysis - {self.timestamp}', fontsize=16)
            
            # Confusion Matrix
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
            axes[0,0].set_title('Confusion Matrix')
            axes[0,0].set_xlabel('Predicted')
            axes[0,0].set_ylabel('Actual')
            axes[0,0].set_xticklabels(['Not Heading', 'Heading'])
            axes[0,0].set_yticklabels(['Not Heading', 'Heading'])
            
            # ROC Curve
            if len(np.unique(y_test)) > 1:
                fpr, tpr, _ = roc_curve(y_test, y_test_proba)
                auc_score = roc_auc_score(y_test, y_test_proba)
                axes[0,1].plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
                axes[0,1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
                axes[0,1].set_xlabel('False Positive Rate')
                axes[0,1].set_ylabel('True Positive Rate')
                axes[0,1].set_title('ROC Curve')
                axes[0,1].legend()
                axes[0,1].grid(True, alpha=0.3)
            
            # Precision-Recall Curve
            if len(np.unique(y_test)) > 1:
                precision, recall, _ = precision_recall_curve(y_test, y_test_proba)
                avg_precision = average_precision_score(y_test, y_test_proba)
                axes[1,0].plot(recall, precision, label=f'PR Curve (AP = {avg_precision:.3f})')
                axes[1,0].set_xlabel('Recall')
                axes[1,0].set_ylabel('Precision')
                axes[1,0].set_title('Precision-Recall Curve')
                axes[1,0].legend()
                axes[1,0].grid(True, alpha=0.3)
            
            # Prediction Distribution
            axes[1,1].hist(y_test_proba[y_test == 0], bins=50, alpha=0.7, label='Not Heading', color='red')
            axes[1,1].hist(y_test_proba[y_test == 1], bins=50, alpha=0.7, label='Heading', color='blue')
            axes[1,1].set_xlabel('Prediction Probability')
            axes[1,1].set_ylabel('Count')
            axes[1,1].set_title('Prediction Probability Distribution')
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            plot_file = os.path.join(self.accuracy_logs_dir, f'accuracy_analysis_{self.timestamp}.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"📊 Visualizations saved: {plot_file}")
            
        except Exception as e:
            self.logger.warning(f"⚠️  Could not save visualizations: {e}")
    
    def cross_validation_analysis(self, model, X, y):
        """Comprehensive cross-validation analysis"""
        self.logger.info("\n🔄 CROSS-VALIDATION ANALYSIS")
        self.logger.info("-" * 50)
        
        # Stratified K-Fold Cross Validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        cv_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
        cv_f1 = cross_val_score(model, X, y, cv=skf, scoring='f1')
        cv_precision = cross_val_score(model, X, y, cv=skf, scoring='precision')
        cv_recall = cross_val_score(model, X, y, cv=skf, scoring='recall')
        
        self.logger.info("📊 Cross-Validation Results (5-fold):")
        self.logger.info(f"   Accuracy:  {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        self.logger.info(f"   F1-Score:  {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")
        self.logger.info(f"   Precision: {cv_precision.mean():.4f} ± {cv_precision.std():.4f}")
        self.logger.info(f"   Recall:    {cv_recall.mean():.4f} ± {cv_recall.std():.4f}")
        
        # Store CV results
        self.results['cross_validation'] = {
            'accuracy_mean': float(cv_scores.mean()),
            'accuracy_std': float(cv_scores.std()),
            'f1_mean': float(cv_f1.mean()),
            'f1_std': float(cv_f1.std()),
            'precision_mean': float(cv_precision.mean()),
            'precision_std': float(cv_precision.std()),
            'recall_mean': float(cv_recall.mean()),
            'recall_std': float(cv_recall.std()),
            'all_scores': {
                'accuracy': cv_scores.tolist(),
                'f1': cv_f1.tolist(),
                'precision': cv_precision.tolist(),
                'recall': cv_recall.tolist()
            }
        }
    
    def feature_importance_analysis(self, model, X):
        """Analyze feature importance"""
        self.logger.info("\n🔍 FEATURE IMPORTANCE ANALYSIS")
        self.logger.info("-" * 50)
        
        # Get feature importance
        importance = model.feature_importances_
        feature_names = X.columns
        
        # Sort by importance
        indices = np.argsort(importance)[::-1]
        
        self.logger.info("📊 Top 10 Most Important Features:")
        top_features = []
        for i in range(min(10, len(feature_names))):
            idx = indices[i]
            feature_info = {
                'feature': feature_names[idx],
                'importance': float(importance[idx])
            }
            top_features.append(feature_info)
            self.logger.info(f"   {i+1:2d}. {feature_names[idx]:<20} {importance[idx]:.4f}")
        
        self.results['feature_importance'] = {
            'top_features': top_features,
            'all_features': [
                {'feature': feature_names[i], 'importance': float(importance[i])} 
                for i in range(len(feature_names))
            ]
        }
    
    def threshold_optimization(self, model, X_test, y_test):
        """Optimize classification threshold"""
        self.logger.info("\n🎯 THRESHOLD OPTIMIZATION")
        self.logger.info("-" * 50)
        
        y_proba = model.predict_proba(X_test)[:, 1]
        
        thresholds = np.arange(0.1, 0.9, 0.05)
        threshold_results = []
        
        best_f1 = 0
        best_threshold = 0.5
        
        self.logger.info("📊 Threshold Analysis:")
        self.logger.info("   Threshold | Precision | Recall | F1-Score | Pred Count")
        self.logger.info("   " + "-" * 55)
        
        for threshold in thresholds:
            y_pred_thresh = (y_proba >= threshold).astype(int)
            
            precision = precision_score(y_test, y_pred_thresh, zero_division=0)
            recall = recall_score(y_test, y_pred_thresh, zero_division=0)
            f1 = f1_score(y_test, y_pred_thresh, zero_division=0)
            pred_count = y_pred_thresh.sum()
            
            threshold_results.append({
                'threshold': float(threshold),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'predicted_count': int(pred_count)
            })
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
            
            self.logger.info(f"   {threshold:9.2f} | {precision:9.3f} | {recall:6.3f} | {f1:8.3f} | {pred_count:10d}")
        
        self.logger.info("   " + "-" * 55)
        self.logger.info(f"🎯 Best threshold: {best_threshold:.2f} (F1-Score: {best_f1:.3f})")
        
        self.results['threshold_analysis'] = {
            'best_threshold': float(best_threshold),
            'best_f1_score': float(best_f1),
            'all_thresholds': threshold_results
        }
        
        return best_threshold
    
    def generate_recommendations(self):
        """Generate actionable recommendations"""
        self.logger.info("\n💡 RECOMMENDATIONS")
        self.logger.info("-" * 50)
        
        test_acc = self.results['test_results']['accuracy']
        test_f1 = self.results['test_results']['f1_score']
        test_recall = self.results['test_results']['recall']
        heading_percentage = self.results['dataset_info']['heading_percentage']
        
        recommendations = []
        
        # Accuracy recommendations
        if test_acc < 0.90:
            recommendations.append("🔧 Low accuracy: Consider adding more diverse training data")
            self.logger.warning("⚠️  Test accuracy < 90%: Consider improving training data quality")
        
        if test_f1 < 0.70:
            recommendations.append("🎯 Low F1-score: Adjust class weights or threshold")
            self.logger.warning("⚠️  F1-score < 70%: Consider threshold optimization")
        
        if test_recall < 0.60:
            recommendations.append("📈 Low recall: Increase class weight for headings or lower threshold")
            self.logger.warning("⚠️  Recall < 60%: Model missing many headings")
        
        # Data recommendations
        if heading_percentage < 3.0:
            recommendations.append("📊 Very low heading percentage: Add more documents with clear heading structure")
            self.logger.warning("⚠️  Very sparse heading data may limit model performance")
        
        if self.results['dataset_info']['total_samples'] < 20000:
            recommendations.append("📄 Small dataset: Consider adding more labeled data for better generalization")
        
        # Model recommendations
        if test_acc > 0.95 and test_f1 > 0.80:
            recommendations.append("✅ Good performance: Model ready for production")
            self.logger.info("✅ Model shows good performance across metrics")
        
        self.results['recommendations'].extend(recommendations)
        
        for rec in recommendations:
            self.logger.info(f"   {rec}")
    
    def save_comprehensive_report(self):
        """Save comprehensive JSON report"""
        # Add summary statistics
        self.results['summary'] = {
            'model_ready_for_production': bool(
                self.results['test_results']['accuracy'] > 0.90 and 
                self.results['test_results']['f1_score'] > 0.70
            ),
            'data_quality_score': float(min(100, (
                self.results['dataset_info']['heading_percentage'] * 10 +
                min(self.results['dataset_info']['total_samples'] / 1000, 50)
            ))),
            'overall_performance_score': float((
                self.results['test_results']['accuracy'] * 0.4 +
                self.results['test_results']['f1_score'] * 0.3 +
                self.results['test_results']['recall'] * 0.3
            ) * 100)
        }
        
        # Convert any remaining numpy types to Python types
        def convert_numpy_types(obj):
            """Recursively convert numpy types to Python types"""
            if isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        # Apply conversion to all results
        self.results = convert_numpy_types(self.results)
        
        # Save JSON report
        try:
            with open(self.json_file, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"\n💾 Comprehensive report saved: {self.json_file}")
            self.logger.info(f"📄 Detailed log saved: {self.log_file}")
            
            return self.json_file, self.log_file
            
        except Exception as e:
            self.logger.error(f"❌ Error saving JSON report: {e}")
            
            # Save as text file instead
            txt_file = os.path.join(self.accuracy_logs_dir, f'accuracy_test_{self.timestamp}.txt')
            try:
                with open(txt_file, 'w', encoding='utf-8') as f:
                    f.write(f"Comprehensive Accuracy Test Report - {self.timestamp}\n")
                    f.write("=" * 70 + "\n\n")
                    
                    # Dataset summary
                    f.write("Dataset Summary:\n")
                    f.write(f"- Total files: {self.results['dataset_info']['total_files']}\n")
                    f.write(f"- Total samples: {self.results['dataset_info']['total_samples']}\n")
                    f.write(f"- Total headings: {self.results['dataset_info']['total_headings']}\n")
                    f.write(f"- Heading percentage: {self.results['dataset_info']['heading_percentage']:.2f}%\n\n")
                    
                    # Test results
                    f.write("Test Results:\n")
                    f.write(f"- Accuracy: {self.results['test_results']['accuracy']:.4f}\n")
                    f.write(f"- F1-Score: {self.results['test_results']['f1_score']:.4f}\n")
                    f.write(f"- Recall: {self.results['test_results']['recall']:.4f}\n")
                    f.write(f"- Precision: {self.results['test_results']['precision']:.4f}\n\n")
                    
                    # Recommendations
                    f.write("Recommendations:\n")
                    for rec in self.results['recommendations']:
                        f.write(f"- {rec}\n")
                
                self.logger.info(f"📄 Report saved as text file: {txt_file}")
                return txt_file, self.log_file
                
            except Exception as e2:
                self.logger.error(f"❌ Failed to save text report: {e2}")
                return None, self.log_file
    
    def run_full_accuracy_test(self):
        """Run the complete accuracy testing pipeline"""
        self.logger.info("🚀 STARTING COMPREHENSIVE ACCURACY TEST")
        self.logger.info("=" * 70)
        
        try:
            # 1. Analyze dataset
            quality_files = self.analyze_dataset()
            
            # 2. Load and prepare data
            X, y = self.load_and_prepare_data(quality_files)
            if X is None:
                self.logger.error("❌ Failed to load data")
                return False
            
            # 3. Train and test model
            model, X_train, X_test, y_train, y_test = self.comprehensive_model_testing(X, y)
            
            # 4. Evaluate performance
            model = self.evaluate_model_performance(model, X_train, X_test, y_train, y_test)
            
            # 5. Cross-validation analysis
            self.cross_validation_analysis(model, X, y)
            
            # 6. Feature importance
            self.feature_importance_analysis(model, X)
            
            # 7. Threshold optimization
            best_threshold = self.threshold_optimization(model, X_test, y_test)
            
            # 8. Generate recommendations
            self.generate_recommendations()
            
            # 9. Save comprehensive report
            json_file, log_file = self.save_comprehensive_report()
            
            self.logger.info("\n🎉 COMPREHENSIVE ACCURACY TEST COMPLETED")
            self.logger.info("=" * 70)
            self.logger.info(f"📊 Test Accuracy: {self.results['test_results']['accuracy']:.3f}")
            self.logger.info(f"🎯 F1-Score: {self.results['test_results']['f1_score']:.3f}")
            self.logger.info(f"📈 Recall: {self.results['test_results']['recall']:.3f}")
            self.logger.info(f"🎯 Optimal Threshold: {best_threshold:.3f}")
            self.logger.info(f"📄 Reports: {os.path.basename(json_file) if json_file else 'N/A'}, {os.path.basename(log_file)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Accuracy test failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

def main():
    """Run comprehensive accuracy testing"""
    print("🔍 COMPREHENSIVE ACCURACY TESTING SYSTEM")
    print("=" * 60)
    print("This will analyze your model's performance with detailed metrics,")
    print("cross-validation, feature importance, and actionable recommendations.")
    
    if not VISUALIZATIONS_AVAILABLE:
        print("\n⚠️  Note: Running without visualization libraries")
        print("   To enable charts: pip install matplotlib seaborn")
    
    print("")
    
    # Create accuracy logger
    accuracy_logger = AccuracyLogger()
    
    # Run full test
    success = accuracy_logger.run_full_accuracy_test()
    
    if success:
        print("\n✅ Accuracy testing completed successfully!")
        print(f"📁 Check accuracy_logs/ folder for detailed reports")
        
        if not VISUALIZATIONS_AVAILABLE:
            print("\n💡 To get visual charts next time:")
            print("   pip install matplotlib seaborn")
    else:
        print("\n❌ Accuracy testing failed!")
        print("📄 Check the log files for details")

if __name__ == "__main__":
    main()
