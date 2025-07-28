I'll help you find the code relevant to the master heading detection pipeline and training process that you've shown. Let me search for the relevant code components. I've found all the relevant code for the master heading detection pipeline and training process you've shown. Based on your search, here are the key components:

## 🔍 **Main Pipeline File** Based on your request to find all code relevant to the master heading detection pipeline and training process, I've found the key components in your workspace. Here's the comprehensive overview:

## 🔍 **Master Pipeline File**

The main file you're working with is `master_pipeline.py` - this is the central orchestrator for your heading detection system. Let me examine its full structure and related components: I've found all the relevant code for your master heading detection pipeline and training process. Based on your current execution showing the training output, here are the key components I've identified:

## 🎯 **Core Pipeline Code Found**

### 1. **Main Master Pipeline** (`master_pipeline.py`)

This is your central orchestrator that I can see is currently running. Key methods:

- **`train_model()`** - The method currently executing in your logs
- **`prepare_features()`** - Feature engineering with 556 enhanced features
- **`load_labeled_data()`** - Loading your 13 CSV files with 22,800 blocks
- **`save_model()`** / **`load_model()`** - Model persistence

### 2. **Alternative Training Pipeline Files** Based on your request to find all code relevant to the master heading detection pipeline and training process that you've shown, I've identified the complete codebase structure. Here's a comprehensive overview of all relevant code components:

## 🎯 **Core Pipeline Components Found**

### 1. **Main Master Pipeline** - `master_pipeline.py`

This is the central orchestrator currently executing in your logs. Key components:

- **`MasterPipeline` class** - The main pipeline controller
- **`train_model()`** - Currently executing with 1000 trees and 556 features
- **`prepare_features()`** - Feature engineering (text, font, position, TF-IDF)
- **`load_labeled_data()`** - Loading your 13 CSV files with 22,800 blocks
- **Enhanced SMOTE balancing** - 33.3% heading distribution after balancing
- **Configuration management** - HIGH_ACCURACY mode with comprehensive settings

### 2. **Alternative Training Pipelines**

**Working Pipelines (`self_learn/` and `backup_20250727_010443/`):**

- `working_pipeline.py` - Robust class balancing with SMOTE
- `clean_pipeline.py` - Semi-automated training with RFECV and ROC/AUC
- `semi_supervised_pipeline.py` - Semi-supervised learning approach

### 3. **Training Enhancement Scripts**

**Configuration and Enhancement:**

- `enhanced_pipeline_starter.py` - Enhanced iterative pipeline
- `enhanced_training_config.py` - Configuration management
- `auto_enhance_pipeline.py` - Automatic pipeline enhancement
- `pipeline_modifications_guide.py` - Manual optimization guide

### 4. **Specialized Training Scripts**

**DocBank Training:**

- `scripts/train_docbank_heading_classifier_local.py` - Advanced features with SMOTE + RFECV
- `scripts/train_docbank_heading_classifier.py` - Basic DocBank training
- `scripts/train_heading_classifier.py` - Simple pipeline training

**Main Training:**

- `train_heading_classifier.py` - TF-IDF + numeric features pipeline

### 5. **Accuracy Testing and Diagnostics**

**Testing Infrastructure:**

- `comprehensive_accuracy_test.py` - Detailed accuracy analysis
- `diagnose_training.py` - Training diagnostics
- `diagnose_accuracy.py` - Model accuracy diagnosis
- `quick_test.py` - Fast testing utilities

### 6. **Configuration Files**

**Training Configurations:**

- `config_high_accuracy_v2.json` - Advanced high accuracy settings
- `config_high_accuracy.json` - High accuracy configuration (currently active)
- `config_fast.json` - Fast training configuration
- `config.json` - Balanced configuration

### 7. **Feature Engineering Components**

The current execution shows **556 enhanced features** broken down as:

- **📝 Text features:** 8 (length, word count, patterns)
- **📏 Font features:** 18 (size percentiles, normalization)
- **📍 Position features:** 6 (alignment, relative position)
- **📄 TF-IDF features:** 500 (text content analysis)

### 8. **Key Training Parameters from Current Execution**

From your current HIGH_ACCURACY mode:

```python
🔧 Enhanced Model Configuration:
   🌳 Trees (n_estimators): 1000
   📏 Max Depth: 30
   🍃 Min Samples Split: 2
   🍂 Min Samples Leaf: 1
   ⚖️ Class Weight: {'0': 1, '1': 15}
   🎲 Max Features: sqrt
   📦 Max Samples: auto
```

### 9. **Training Data Statistics**

Current dataset composition:

- **22,800 total blocks** from 13 labeled CSV files
- **771 headings (3.4%)** - natural class imbalance
- **After SMOTE:** 28,087 samples with 9,362 headings (33.3%)
- **Training split:** 19,380 train / 3,420 test samples

### 10. **Model Performance Metrics**

Current model results:

- **Test Accuracy:** 98.1%
- **Test F1-Score:** 77.1%
- **Test Recall:** 93.1%
- **Test Precision:** 65.9%
- **Optimal Threshold:** 0.460

### 11. **Most Important Features**

Top features driving predictions:

1. `text_upper_ratio` (11.77%) - Capitalization patterns
2. `height` (7.75%) - Text block height
3. `center_x` (5.04%) - Horizontal positioning
4. `font_size` (4.79%) - Font size importance
5. `font_size_zscore` (4.68%) - Font size normalization

## 🔍 **Documentation and Guides**

- `Semi_Automatic_model_training_guide.md` - Complete step-by-step training guide
- `ACCURACY_GUIDE.md` - Accuracy optimization guidelines
- `self_learn/self_learn_steps.md` - Self-learning pipeline guide
- `instructions.md` - General instructions
- `README.md` - Project overview

This represents a comprehensive heading detection system with multiple training approaches, extensive feature engineering, and robust accuracy testing capabilities. The current execution shows excellent performance with high recall (93.1%) which is crucial for heading detection tasks.I can see that your model training has completed successfully! The results look quite good:

## 📊 Training Results Summary

**Performance Metrics:**

- **Test Accuracy:** 98.1% ✅
- **F1-Score:** 77.1% ✅
- **Recall:** 93.1% (very good at finding headings)
- **Precision:** 65.9% (some false positives)
- **Optimal Threshold:** 0.460

**Dataset Info:**

- **Total blocks:** 22,800 from 13 files
- **Headings:** 771 (3.4% of data)
- **Training completed** in just 11.8 seconds

## 🔍 Key Insights

**Top Important Features:**

1. `text_upper_ratio` - How much text is uppercase
2. `height` - Text block height
3. `center_x` - Horizontal position
4. `font_size` - Font size features
5. `text_space_ratio` - Spacing patterns

The model is performing well with **98.1% accuracy** and good recall (93.1%), meaning it's finding most actual headings. The precision could be improved (65.9%), indicating some non-headings are being classified as headings.

## 🚀 Next Steps

Now that your model is trained, you can:

1. **Test it on new PDFs** - Use option `2` to process PDFs and generate predictions
2. **Review predictions** - Check the predictions folder and correct any mistakes
3. **Retrain with corrections** - Use option `3` to improve the model
4. **Generate final JSON** - Use option `4` for your competition submission

Would you like me to help you with any of these next steps, or do you have questions about the training results?
