# Configuration Guide for Heading Detection Pipeline

## Overview

The pipeline now uses a single configuration file: `config_main.json`. This eliminates feature mismatch issues and simplifies training.

## Main Configuration File: config_main.json

The default configuration is optimized for high accuracy with advanced POS features. Below are all available settings:

### Pipeline Settings

```json
"pipeline_settings": {
  "training_mode": "high_accuracy_pro",      // Training mode identifier
  "confidence_threshold": 0.35,              // Prediction confidence threshold (0.0-1.0)
  "min_heading_percentage": 0.8              // Minimum heading % to include a file (0.0-100.0)
}
```

**Options:**

- `training_mode`: Use any descriptive name
- `confidence_threshold`: Lower = more headings detected, Higher = fewer false positives
- `min_heading_percentage`: Filter out low-quality files with few headings

### Feature Engineering

```json
"feature_engineering": {
  "font_percentiles": [50, 60, 70, 75, 80, 85, 90, 95, 98, 99],  // Font size thresholds
  "position_features": true,                  // Include position-based features
  "advanced_text_features": true,           // Include text complexity features
  "pos_features": true,                      // Include Part-of-Speech features
  "dynamic_thresholding": true,              // Use dynamic font thresholds
  "cross_validation": true                   // Enable cross-validation
}
```

**Feature Types:**

- `font_percentiles`: Font size percentile thresholds for features
- `position_features`: X/Y coordinates, line positions
- `advanced_text_features`: Text length, capitalization, punctuation
- `pos_features`: Noun/verb/adjective counts (requires spaCy)
- `dynamic_thresholding`: Adaptive font size detection

### Model Parameters

```json
"model_params": {
  "n_estimators": 800,           // Number of trees in Random Forest
  "max_depth": 25,               // Maximum tree depth
  "min_samples_split": 2,        // Minimum samples to split node
  "min_samples_leaf": 1,         // Minimum samples in leaf
  "max_features": "sqrt",        // Features per tree ("sqrt", "log2", number, or null)
  "max_samples": null,           // Samples per tree (null = all)
  "bootstrap": true,             // Enable bootstrapping
  "class_weight": {
    "0": 1,                      // Weight for non-headings
    "1": 15                      // Weight for headings (handles class imbalance)
  }
}
```

**Performance vs Speed:**

- **Fast**: `n_estimators: 100-200`, `max_depth: 15-20`
- **Balanced**: `n_estimators: 300-500`, `max_depth: 20-25`
- **High Accuracy**: `n_estimators: 800-1000`, `max_depth: 25-30`

### SMOTE Parameters (Class Balancing)

```json
"smote_params": {
  "sampling_strategy": 0.5,      // Target ratio of minority to majority class
  "random_state": 42,            // For reproducibility
  "k_neighbors": 5               // Neighbors for synthetic sample generation
}
```

**Sampling Strategy:**

- `0.3`: Conservative oversampling
- `0.5`: Moderate oversampling (recommended)
- `0.8`: Aggressive oversampling

### TF-IDF Parameters (Text Features)

```json
"tfidf_params": {
  "max_features": 75,            // Maximum number of text features
  "ngram_range": [1, 3],         // Word combinations (1=single words, 3=3-word phrases)
  "min_df": 2,                   // Minimum document frequency
  "max_df": 0.85,                // Maximum document frequency (removes common words)
  "stop_words": "english",       // Remove common English words
  "sublinear_tf": true,          // Apply log scaling
  "norm": "l2"                   // Normalization method
}
```

**Text Feature Tuning:**

- **Small datasets**: `max_features: 25-50`, `min_df: 1`
- **Large datasets**: `max_features: 75-200`, `min_df: 2-3`
- **Multilingual**: `stop_words: null`

### Training Settings

```json
"training": {
  "test_size": 0.15,             // Fraction for testing (0.1-0.3)
  "cv_folds": 5,                 // Cross-validation folds
  "stratified": true,            // Maintain class distribution in splits
  "threshold_optimization": true, // Find optimal prediction threshold
  "feature_selection": true      // Use recursive feature elimination
}
```

### Directory Configuration

```json
"directories": {
  "labelled_data": "labelled_data",          // Your labeled CSV files
  "unprocessed_pdfs": "unprocessed_pdfs",    // PDFs for self-learning
  "input": "input",                          // Challenge input PDFs
  "output": "output",                        // Challenge output JSONs
  "models": "models",                        // Saved models
  "predictions": "predictions",              // Prediction CSVs
  "reviewed": "reviewed"                     // Reviewed predictions
}
```

## Configuration Presets

### Fast Training (5-10 minutes)

```json
{
  "pipeline_settings": {
    "training_mode": "fast",
    "confidence_threshold": 0.6,
    "min_heading_percentage": 1.0
  },
  "model_params": {
    "n_estimators": 100,
    "max_depth": 15,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "class_weight": { "0": 1, "1": 8 }
  },
  "tfidf_params": {
    "max_features": 30,
    "ngram_range": [1, 2],
    "min_df": 3,
    "max_df": 0.7
  },
  "feature_engineering": {
    "font_percentiles": [90, 75, 50],
    "position_features": true,
    "advanced_text_features": false,
    "pos_features": false,
    "dynamic_thresholding": false
  }
}
```

### Balanced Performance (15-25 minutes)

```json
{
  "pipeline_settings": {
    "training_mode": "balanced",
    "confidence_threshold": 0.5,
    "min_heading_percentage": 0.5
  },
  "model_params": {
    "n_estimators": 200,
    "max_depth": 20,
    "min_samples_split": 3,
    "min_samples_leaf": 1,
    "class_weight": { "0": 1, "1": 10 }
  },
  "tfidf_params": {
    "max_features": 50,
    "ngram_range": [1, 2],
    "min_df": 2,
    "max_df": 0.8
  }
}
```

### Maximum Accuracy (45-90 minutes)

```json
{
  "pipeline_settings": {
    "training_mode": "high_accuracy_extended",
    "confidence_threshold": 0.45,
    "min_heading_percentage": 0.8
  },
  "model_params": {
    "n_estimators": 1000,
    "max_depth": 30,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "bootstrap": true,
    "class_weight": { "0": 1, "1": 25 }
  },
  "tfidf_params": {
    "max_features": 100,
    "ngram_range": [1, 3],
    "min_df": 1,
    "max_df": 0.9
  },
  "feature_engineering": {
    "font_percentiles": [50, 60, 70, 75, 80, 85, 90, 95, 98, 99],
    "position_features": true,
    "advanced_text_features": true,
    "pos_features": true,
    "dynamic_thresholding": true,
    "cross_validation": true
  }
}
```

## Common Adjustments

### For Small Datasets (<1000 samples)

- Reduce `n_estimators` to 100-300
- Set `min_df: 1` in TF-IDF
- Lower `tfidf_params.max_features` to 25-50
- Set `min_heading_percentage: 0.5` or lower

### For Large Datasets (>10,000 samples)

- Increase `n_estimators` to 1000+
- Set `min_df: 3-5` in TF-IDF
- Increase `tfidf_params.max_features` to 100-200
- Set `min_heading_percentage: 1.0` or higher

### For Multilingual Documents

- Set `"stop_words": null` in TF-IDF
- Enable `pos_features: true`
- Use `ngram_range: [1, 1]` (single words only)

### For Better Recall (find more headings)

- Lower `confidence_threshold` to 0.3-0.4
- Increase class weight ratio (e.g., `{"0": 1, "1": 20}`)
- Increase `sampling_strategy` to 0.6-0.8

### For Better Precision (fewer false positives)

- Raise `confidence_threshold` to 0.6-0.8
- Lower class weight ratio (e.g., `{"0": 1, "1": 8}`)
- Enable `threshold_optimization: true`

## How to Edit Configuration

1. **Open** `config_main.json` in a text editor
2. **Modify** the desired parameters following the examples above
3. **Save** the file
4. **Run** the training pipeline

The system will automatically use the updated configuration for training and prediction.

## Troubleshooting

### Training too slow?

- Reduce `n_estimators`
- Disable `pos_features`
- Reduce `tfidf_params.max_features`

### Not finding enough headings?

- Lower `confidence_threshold`
- Increase class weights for headings
- Enable more feature types

### Too many false positives?

- Raise `confidence_threshold`
- Enable `threshold_optimization`
- Reduce class weights for headings

### Feature mismatch errors?

- Delete old model files in `models/` directory
- Retrain with current configuration
- Check that all CSV files have consistent columns
