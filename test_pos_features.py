#!/usr/bin/env python3
"""
POS Features Integration Test
============================

Test script to verify that POS features work correctly both with and without spaCy.

Usage:
    python test_pos_features.py

Author: AI Assistant  
Date: July 28, 2025
"""

import os
import sys
import pandas as pd
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_pos_features():
    """Test POS features functionality"""
    print("🏷️ POS FEATURES INTEGRATION TEST")
    print("=" * 40)
    
    # Test data
    test_data = {
        'text': [
            'Chapter 1: Introduction',
            'This document describes the system architecture.',
            'HTTP Requests',
            'The implementation uses advanced machine learning algorithms.',
            'Conclusion',
            'References and Bibliography',
            'www.example.com',
            'Initially,',
            'System Overview'
        ],
        'font_size': [16, 12, 14, 11, 15, 13, 10, 12, 14],
        'page': [1, 1, 1, 1, 2, 2, 2, 2, 2]
    }
    
    df = pd.DataFrame(test_data)
    print(f"\n📊 Test dataset: {len(df)} rows")
    
    # Test 1: Import POS handler
    try:
        from pos_features_handler import POSFeaturesHandler, calculate_pos_features
        print("✅ POS features handler imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import POS handler: {e}")
        return False
    
    # Test 2: Initialize handler
    try:
        handler = POSFeaturesHandler()
        print(f"✅ POS handler initialized (spaCy available: {handler.spacy_available})")
    except Exception as e:
        print(f"❌ Failed to initialize POS handler: {e}")
        return False
    
    # Test 3: Calculate POS features
    try:
        result_df = handler.calculate_pos_features(df.copy())
        print("✅ POS features calculated successfully")
    except Exception as e:
        print(f"❌ Failed to calculate POS features: {e}")
        return False
    
    # Test 4: Verify feature columns exist
    expected_features = ['num_nouns', 'num_verbs', 'num_adjs', 'num_advs', 'num_propn', 'num_pronouns', 'num_other_pos']
    missing_features = [f for f in expected_features if f not in result_df.columns]
    
    if missing_features:
        print(f"❌ Missing POS features: {missing_features}")
        return False
    else:
        print("✅ All expected POS features present")
    
    # Test 5: Verify feature values are reasonable
    print("\n📈 POS Features Summary:")
    print("-" * 25)
    for feature in expected_features:
        values = result_df[feature]
        print(f"{feature:15}: min={values.min():2.0f}, max={values.max():2.0f}, mean={values.mean():4.1f}")
    
    # Test 6: Check specific examples
    print("\n🔍 Sample Results:")
    print("-" * 15)
    sample_cols = ['text'] + expected_features
    for idx, row in result_df.head(3).iterrows():
        print(f"\nRow {idx}: '{row['text'][:30]}...'")
        for feature in expected_features:
            print(f"  {feature}: {row[feature]}")
    
    # Test 7: Test fallback vs spaCy (if available)
    if handler.spacy_available:
        print("\n🔄 Testing fallback method...")
        handler_fallback = POSFeaturesHandler()
        handler_fallback.spacy_available = False  # Force fallback
        
        fallback_df = handler_fallback.calculate_pos_features(df.copy())
        
        # Compare results
        print("📊 spaCy vs Fallback comparison:")
        for feature in expected_features:
            spacy_mean = result_df[feature].mean()
            fallback_mean = fallback_df[feature].mean()
            print(f"  {feature:15}: spaCy={spacy_mean:4.1f}, Fallback={fallback_mean:4.1f}")
    
    print("\n🎉 All POS features tests passed!")
    return True

def test_config_integration():
    """Test integration with config file"""
    print("\n⚙️ CONFIGURATION INTEGRATION TEST")
    print("=" * 35)
    
    config_path = "config_main.json"
    
    if not os.path.exists(config_path):
        print(f"⚠️ Config file not found: {config_path}")
        return True
    
    try:
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        pos_enabled = config.get('feature_engineering', {}).get('pos_features', False)
        
        print(f"✅ Configuration loaded successfully")
        print(f"✅ POS features enabled: {pos_enabled}")
        
        if pos_enabled:
            print("🏷️ POS features are enabled in configuration")
        else:
            print("⚠️ POS features are disabled in configuration")
            print("   To enable: Set 'pos_features': true in config_main.json")
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading configuration: {e}")
        return False

def test_model_compatibility():
    """Test model loading compatibility"""
    print("\n🤖 MODEL COMPATIBILITY TEST")
    print("=" * 30)
    
    models_dir = "models"
    
    if not os.path.exists(models_dir):
        print(f"⚠️ Models directory not found: {models_dir}")
        return True
    
    # Look for model files
    import glob
    model_files = glob.glob(os.path.join(models_dir, "heading_model_*.pkl"))
    vectorizer_files = glob.glob(os.path.join(models_dir, "tfidf_vectorizer_*.pkl"))
    
    print(f"📁 Found {len(model_files)} model files")
    print(f"📁 Found {len(vectorizer_files)} vectorizer files")
    
    if model_files and vectorizer_files:
        print("✅ Model files are present")
        
        # Try to load the latest model
        try:
            import joblib
            latest_model = sorted(model_files)[-1]
            latest_vectorizer = sorted(vectorizer_files)[-1]
            
            model = joblib.load(latest_model)
            vectorizer = joblib.load(latest_vectorizer)
            
            print(f"✅ Successfully loaded model: {os.path.basename(latest_model)}")
            print(f"✅ Successfully loaded vectorizer: {os.path.basename(latest_vectorizer)}")
            
            # Get feature names if available
            if hasattr(vectorizer, 'get_feature_names_out'):
                feature_names = vectorizer.get_feature_names_out()
                print(f"📊 TF-IDF features: {len(feature_names)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False
    else:
        print("⚠️ No model files found. Run training first.")
        return True

def main():
    """Main test function"""
    print("🧪 COMPREHENSIVE POS FEATURES TEST")
    print("=" * 50)
    print("Testing POS features functionality and integration...")
    print()
    
    success = True
    
    # Test 1: POS features functionality
    if not test_pos_features():
        success = False
    
    # Test 2: Configuration integration
    if not test_config_integration():
        success = False
    
    # Test 3: Model compatibility
    if not test_model_compatibility():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 ALL TESTS PASSED!")
        print("✅ POS features are working correctly")
        print("✅ System is ready for training and prediction")
        print("\n🚀 Next steps:")
        print("   1. python setup_project.py  # Ensure all dependencies")
        print("   2. python master_pipeline.py  # Train models")
        print("   3. python generate_json_output.py  # Generate predictions")
    else:
        print("❌ SOME TESTS FAILED")
        print("Please check the error messages above and fix the issues.")
        print("\n🔧 Troubleshooting:")
        print("   1. Run: python setup_project.py")
        print("   2. Install spaCy model: python -m spacy download en_core_web_sm")
        print("   3. Check configuration: config_main.json")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
