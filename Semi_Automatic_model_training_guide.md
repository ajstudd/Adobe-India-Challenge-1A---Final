# Complete Step-by-Step Guide: PDF Heading Detection Pipeline

## 🎯 Overview

This guide will help you train a machine learning model to extract document outlines (headings) from PDFs and generate JSON output files. The pipeline uses a semi-supervised approach to continuously improve accuracy.

---

## 📋 Prerequisites & Setup

### Step 1: Verify Your Environment

1. **Check Python version** (must be 3.8+):

   ```cmd
   python --version
   ```

2. **Navigate to project directory**:

   ```cmd
   cd c:\Users\j7654\WorkStation\Adobe-India-Hackathon-25\Adobe-India-Challenge-1A
   ```

3. **Install required packages**:
   ```cmd
   pip install -r requirements.txt
   ```

### Step 2: Understand the Folder Structure

Your project should have these folders:

- `📁 labelled_data/` - Training data (CSV files with headings marked)
- `📁 unprocessed_pdfs/` - PDFs to analyze and predict headings
- `📁 input/` - Final PDFs for JSON generation (competition submission)
- `📁 output/` - Generated JSON files (final results)
- `📁 models/` - Saved ML models (auto-created)
- `📁 predictions/` - Predicted headings for review (auto-created)
- `📁 reviewed/` - Your corrected predictions (auto-created)

---

## 🚀 Phase 1: Initial Model Training

### Step 3: Test Your Configuration

First, let's verify everything is working:

```cmd
python quick_test.py
```

This will show you:

- Available configuration modes
- Training data statistics
- Expected training times

### Step 4: Choose Your Training Mode

Run the main pipeline:

```cmd
python master_pipeline.py
```

**You'll see a menu like this:**

```
🤖 MASTER HEADING DETECTION PIPELINE
====================================
⚙️  Current mode: BALANCED
0. ⚙️  Change configuration mode
1. 🌱 Train new model on labeled data
2. 📄 Process PDFs (generate predictions)
...
```

**First, choose option `0` to select your training mode:**

**For beginners**: Choose option `1` - Fast Training (5-10 min)

- Good for testing and learning the system
- 100 trees, basic features

**For best results**: Choose option `3` - High Accuracy (30-45 min)

- Best possible accuracy
- 1000 trees, advanced features

### Step 5: Train Your First Model

1. **Select option `1`** from the main menu: "🌱 Train new model on labeled data"

2. **The system will:**

   - Load all CSV files from labelled_data folder
   - Show statistics about your training data
   - Train a Random Forest model
   - Test the model and show accuracy metrics

3. **You'll see output like:**
   ```
   📊 Loading labeled training data...
   📄 Found 25 labeled CSV files
   ✅ document1.csv: 156 blocks, 12 headings (7.7%)
   ✅ document2.csv: 203 blocks, 18 headings (8.9%)
   ...
   🌳 Training heading detection model...
   ✅ Model training completed!
   📈 Test Accuracy: 0.943
   🎯 Test F1-Score: 0.876
   💾 Model saved: models/heading_model_latest.pkl
   ```

---

## 📄 Phase 2: Generate Predictions

### Step 6: Add PDFs for Analysis

1. **Place PDF files** in the unprocessed_pdfs folder
   - These are PDFs where you want to detect headings
   - The model will predict which text blocks are headings

### Step 7: Generate Predictions

1. **Select option `2`** from the main menu: "📄 Process PDFs (generate predictions)"

2. **When asked "Max PDFs to process"**:

   - Press Enter to process all PDFs
   - Or type a number like `5` to limit processing

3. **The system will:**

   - Extract text blocks from each PDF
   - Predict which blocks are headings
   - Save predictions as CSV files in predictions folder

4. **You'll see output like:**
   ```
   📄 Processing PDF batch for predictions
   --- Processing PDF 1/5: research_paper ---
   ✅ Extracted 234 blocks using PyMuPDF fallback
   ✅ Predicted 18 headings out of 234 blocks
   💾 Saved predictions: predictions/research_paper_predictions.csv
   ```

---

## 🔍 Phase 3: Review and Improve (Semi-Supervised Learning)

### Step 8: Manual Review Process

1. **Navigate to the predictions folder**
2. **Open CSV files** in Excel, LibreOffice, or any spreadsheet editor
3. **Look at the `is_heading` column**:

   - `1` = Model predicted this is a heading
   - `0` = Model predicted this is NOT a heading

4. **Correct the predictions**:

   - If text should be a heading but shows `0`, change it to `1`
   - If text is NOT a heading but shows `1`, change it to `0`
   - Focus on obvious mistakes first

5. **Save corrected files** to the reviewed folder
   - Keep the same filename (e.g., `research_paper_predictions.csv`)

**Example corrections:**

```csv
text,x,y,font_size,is_heading,heading_probability
"Introduction",50,100,16,1,0.85          ← Correct (large font = heading)
"Page 1 of 10",400,50,8,0,0.12           ← Correct (footer = not heading)
"3.2 Results",50,200,14,0,0.45           ← WRONG! Change to 1 (clearly a heading)
"See Figure 1 below",50,300,12,1,0.65    ← WRONG! Change to 0 (not a heading)
```

### Step 9: Retrain with Corrections

1. **Select option `3`** from the main menu: "🔄 Retrain with manual corrections"

2. **The system will:**

   - Load your original training data
   - Load your corrected predictions from reviewed folder
   - Combine both datasets
   - Train a new, improved model

3. **You'll see output like:**
   ```
   🔄 RETRAINING WITH CORRECTIONS
   ✅ Loaded research_paper_predictions.csv: 234 blocks, 22 headings
   ✅ Loaded thesis_predictions.csv: 189 blocks, 15 headings
   📊 Combined training data: 1,450 blocks
   🌳 Training heading detection model...
   ✅ Retraining completed! (Cycle 1)
   ```

---

## 🔄 Phase 4: Iterative Improvement

### Step 10: Repeat the Cycle

**For maximum accuracy, repeat steps 6-9 multiple times:**

1. **Add more PDFs** to unprocessed_pdfs
2. **Generate predictions** (option 2)
3. **Review and correct** predictions
4. **Retrain model** (option 3)

**Each cycle improves your model!**

### Step 11: Monitor Progress

**Select option `6`** to see pipeline status:

- Current model accuracy
- Number of training cycles completed
- Files processed vs reviewed

---

## 📤 Phase 5: Final JSON Generation

### Step 12: Prepare for Final Output

1. **Place final PDFs** in the input folder
   - These are the PDFs for your competition submission
   - Or use PDFs already in unprocessed_pdfs

### Step 13: Generate JSON Files

1. **Select option `4`** from the main menu: "📤 Generate JSON output"

2. **When asked "Process 'input' folder?"**:

   - Type `y` to process PDFs from input folder
   - Type `n` to process PDFs from unprocessed_pdfs folder

3. **The system will:**

   - Extract text from each PDF
   - Predict headings using your trained model
   - Generate JSON files in the required format
   - Save JSON files to output folder

4. **You'll see output like:**
   ```
   📤 GENERATING JSON OUTPUT
   --- Generating JSON for: research_paper ---
   ✅ Generated JSON with 18 headings
   💾 Saved: output/research_paper.json
   ```

### Step 14: Verify JSON Output

**Check the output folder for JSON files like:**

```json
{
  "title": "Research Paper",
  "outline": [
    {
      "level": "H1",
      "text": "Introduction",
      "page": 1
    },
    {
      "level": "H2",
      "text": "Background and Related Work",
      "page": 2
    },
    {
      "level": "H3",
      "text": "3.1 Previous Studies",
      "page": 3
    }
  ]
}
```

---

## 🎯 Quick Start Workflow (Automated)

### Step 15: Full Automated Cycle

**For a complete end-to-end run, select option `5`**: "🚀 Run full cycle (all steps)"

This will:

1. ✅ Train/load model
2. 📄 Process all PDFs
3. ⏸️ **Pause for your review** (optional)
4. 🔄 Retrain with corrections
5. 📤 Generate final JSON output

**When prompted "Wait for manual review?"**:

- Type `y` to pause and let you review predictions
- Type `n` to skip review and use current model

---

## 💡 Tips for Maximum Accuracy

### Configuration Tips

1. **Start with Fast mode** to test everything
2. **Switch to High Accuracy mode** for final training:

   ```
   Select option 0 → option 3 → High Accuracy
   ```

3. **Monitor training times**:
   - Fast: 5-10 minutes
   - Balanced: 15-20 minutes
   - High Accuracy: 30-45 minutes

### Review Tips

1. **Focus on obvious mistakes** first:

   - Large text that should be headings
   - Small text (footers, captions) marked as headings

2. **Look for patterns**:

   - Section numbers (1.1, 2.3, etc.)
   - Bold or larger fonts
   - Text at the start of sections

3. **Quality over quantity**:
   - Better to correct 50 predictions perfectly than 200 hastily

### Model Improvement Tips

1. **Add diverse training data**:

   - Different document types
   - Various PDF formats
   - Multiple languages if needed

2. **Review at least 2-3 cycles**:
   - Each cycle significantly improves accuracy
   - Model learns from your corrections

---

## 🔧 Troubleshooting

### Common Issues

**"No labeled CSV files found"**:

- Ensure labelled_data folder contains CSV files
- CSV files must have `is_heading` and `text` columns

**"No PDF files found"**:

- Check that PDFs are in correct folder (unprocessed_pdfs or input)
- Ensure files have `.pdf` extension

**Training takes too long**:

- Switch to Fast configuration (option 0 → option 1)
- Reduce number of PDFs processed

**Low accuracy**:

- Add more labeled training data
- Use High Accuracy configuration
- Complete more review cycles

### Getting Help

**Check pipeline status**: Select option `6`
**View configuration**: Select option `8`
**Clean directories**: Select option `7` if you need to start fresh

---

## 🎉 Success Criteria

**You'll know you're successful when:**

1. ✅ Model training completes without errors
2. ✅ Test accuracy > 90% (shown after training)
3. ✅ Predictions look reasonable when reviewed
4. ✅ JSON files are generated in correct format
5. ✅ Each review cycle improves the model

**Final deliverable**: JSON files in output folder matching the required schema for competition submission.

---

**🚀 You're now ready to run the complete pipeline! Start with Step 3 and follow each step carefully. The semi-supervised approach will help you achieve high accuracy through iterative improvement.**
