# Adobe India Hackathon 2025 - Challenge 1A

## Simplified Heading Detection Pipeline

### 🚀 Quick Start

**Run the master pipeline:**

```bash
python master_pipeline.py
```

This will show you an interactive menu with all options.

**For simple retraining:**

```bash
python retrain_simple.py
```

### ⚙️ Configuration

**Single Configuration File**: `config_main.json`

- All settings in one place
- No more feature mismatch issues
- See `CONFIG_README.md` for customization options

### 📁 Directory Structure

```
📂 Adobe-India-Challenge-1A/
├── 📄 master_pipeline.py       # ⭐ THE MAIN SCRIPT
├── 📄 retrain_simple.py        # 🔄 SIMPLE RETRAINING
├── 📄 config_main.json         # ⚙️  MAIN CONFIGURATION
├── 📄 CONFIG_README.md         # 📖 CONFIGURATION GUIDE
├── 📁 labelled_data/           # Your training CSV files
├── 📁 unprocessed_pdfs/        # PDFs to process for predictions
├── 📁 input/                   # Final PDFs for competition submission
├── 📁 output/                  # Final JSON outputs
├── 📁 predictions/             # Generated predictions (for review)
├── 📁 reviewed/                # Manually corrected predictions
├── 📁 models/                  # Trained models
├── 📁 old_configs_backup/      # Backed up old config files
└── 📁 src/, scripts/           # Supporting extraction code
```

### 🔄 Complete Workflow

1. **Train Model**: Uses your labeled CSV data in `labelled_data/`
2. **Process PDFs**: Extracts blocks and predicts headings
3. **Manual Review**: Review/correct predictions in `predictions/` folder
4. **Retrain**: Improve model with your corrections
5. **JSON Output**: Generate final competition JSON files

### 📊 Your Data

- **15 labeled CSV files** in `labelled_data/`
- **40+ unprocessed PDFs** in `unprocessed_pdfs/`
- All connected through the master pipeline!

### 🎯 Key Features

✅ **Simplified Configuration**: Single `config_main.json` file  
✅ **No Feature Mismatch**: Consistent training and prediction  
✅ **Semi-Automated**: Train → Predict → Review → Retrain cycle  
✅ **ML-Powered**: Random Forest + SMOTE + TF-IDF + POS features  
✅ **JSON Output**: Ready for competition submission  
✅ **Flexible**: Can process individual PDFs or batches

## Docker Deployment

### How to Build

```bash
docker build --platform linux/amd64 -t adobe_1a_backend .
```

### How to Run

```bash
docker run --rm -e MODE=1A -v <absolute_input_path>:/app/input:ro -v <absolute_output_path>:/app/output --network none adobe_1a_backend
```
