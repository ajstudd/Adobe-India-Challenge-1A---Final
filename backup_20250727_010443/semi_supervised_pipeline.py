"""
Semi-Supervised Learning Pipeline
================================

This pipeline:
1. Trains model on existing labeled data
2. Processes all PDFs in unprocessed_pdfs/ folder
3. Generates predictions and saves to predictions/ folder
4. Waits for your manual review/corrections
5. Retrains model using original + corrected data
6. Repeats the cycle for continuous improvement
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
import time

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the working pipeline and PDF extractor
from working_pipeline import WorkingHeadingDetectionPipeline
from pdf_extractor import PDFExtractor

class SemiSupervisedPipeline:
    """Semi-supervised learning pipeline for heading detection"""
    
    def __init__(self):
        self.base_dir = os.path.dirname(__file__)
        self.project_root = os.path.abspath(os.path.join(self.base_dir, '..'))
        
        # Directories
        self.labelled_data_dir = os.path.join(self.project_root, 'labelled_data')
        self.unprocessed_pdfs_dir = os.path.join(self.project_root, 'unprocessed_pdfs')
        self.predictions_dir = os.path.join(self.base_dir, 'predictions')
        self.reviewed_dir = os.path.join(self.base_dir, 'reviewed')
        self.models_dir = os.path.join(self.base_dir, 'models')
        
        # Create directories
        for dir_path in [self.predictions_dir, self.reviewed_dir, self.models_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Pipeline components
        self.heading_pipeline = WorkingHeadingDetectionPipeline()
        self.pdf_extractor = PDFExtractor()
        
        # Cycle tracking
        self.current_cycle = 0
    
    def step1_initial_training(self):
        """Step 1: Train initial model on existing labeled data"""
        logger.info("\\n" + "="*70)
        logger.info("🚀 STEP 1: INITIAL TRAINING ON LABELED DATA")
        logger.info("="*70)
        
        # Load existing labeled data
        df = self.heading_pipeline.load_labeled_data(min_heading_percentage=1.0)
        if df is None:
            logger.error("❌ No quality labeled data found")
            return False
        
        # Train model
        logger.info("🌳 Training initial model...")
        results = self.heading_pipeline.train_model(df)
        
        # Save model
        model_path = self.heading_pipeline.save_model(f"cycle_0_initial")
        
        logger.info("✅ Step 1 completed successfully!")
        logger.info(f"📈 Test accuracy: {results['test_accuracy']:.3f}")
        logger.info(f"🎯 Optimal threshold: {results['optimal_threshold']:.3f}")
        logger.info(f"📊 Predicted {results['predicted_headings']}/{results['total_test_samples']} headings in test")
        
        return True
    
    def step2_process_all_pdfs(self, max_pdfs=None):
        """Step 2: Process ALL PDFs in unprocessed_pdfs folder"""
        logger.info("\\n" + "="*70)
        logger.info("📄 STEP 2: PROCESSING ALL PDFs FOR PREDICTIONS")
        logger.info("="*70)
        
        # Check if model exists
        if not self.heading_pipeline.load_model():
            logger.error("❌ No trained model found. Run step1_initial_training() first.")
            return False
        
        # Find all PDF files
        pdf_files = list(Path(self.unprocessed_pdfs_dir).glob("*.pdf"))
        
        if not pdf_files:
            logger.warning("⚠️  No PDF files found in unprocessed_pdfs directory")
            return True
        
        if max_pdfs:
            pdf_files = pdf_files[:max_pdfs]
        
        logger.info(f"📄 Found {len(pdf_files)} PDF files to process")
        
        # Process each PDF
        successful_predictions = 0
        failed_predictions = 0
        
        for i, pdf_path in enumerate(pdf_files):
            pdf_name = pdf_path.stem
            logger.info(f"\\n--- Processing PDF {i+1}/{len(pdf_files)}: {pdf_name} ---")
            
            try:
                # Extract PDF to CSV
                csv_path = os.path.join(self.predictions_dir, f"{pdf_name}_extracted.csv")
                df_extracted = self.pdf_extractor.extract_pdf_to_blocks(str(pdf_path), csv_path)
                
                if df_extracted is None or len(df_extracted) == 0:
                    logger.error(f"❌ Failed to extract blocks from {pdf_name}")
                    failed_predictions += 1
                    continue
                
                logger.info(f"✅ Extracted {len(df_extracted)} blocks from {pdf_name}")
                
                # Generate predictions
                predictions_path = self.generate_predictions_for_csv(csv_path, pdf_name)
                
                if predictions_path:
                    successful_predictions += 1
                    logger.info(f"💾 Predictions saved: {predictions_path}")
                else:
                    failed_predictions += 1
                
            except Exception as e:
                logger.error(f"❌ Error processing {pdf_name}: {e}")
                failed_predictions += 1
        
        logger.info(f"\\n🎯 Processing complete!")
        logger.info(f"✅ Successfully processed: {successful_predictions} PDFs")
        logger.info(f"❌ Failed: {failed_predictions} PDFs")
        logger.info(f"📁 Predictions saved to: {self.predictions_dir}")
        
        return successful_predictions > 0
    
    def generate_predictions_for_csv(self, csv_path, pdf_name):
        """Generate predictions for a single CSV file"""
        try:
            # Load extracted CSV
            df = pd.read_csv(csv_path)
            
            if 'text' not in df.columns:
                logger.error(f"❌ CSV missing 'text' column: {csv_path}")
                return None
            
            # Prepare features
            X = self.heading_pipeline.prepare_features(df)
            
            if X is None:
                logger.error(f"❌ Failed to prepare features for {pdf_name}")
                return None
            
            # Make predictions
            predictions, probabilities = self.heading_pipeline.predict_with_threshold(X)
            
            # Add prediction columns
            df['predicted_heading'] = predictions
            df['heading_confidence'] = probabilities
            df['is_heading'] = 0  # Default to 0 - user will correct this
            df['user_reviewed'] = 0  # Track if user has reviewed this row
            df['prediction_cycle'] = self.current_cycle
            
            # Add metadata
            df['pdf_source'] = pdf_name
            df['prediction_date'] = datetime.now().isoformat()
            
            # Save predictions
            predictions_path = os.path.join(self.predictions_dir, f"{pdf_name}_predictions.csv")
            df.to_csv(predictions_path, index=False)
            
            # Log statistics
            predicted_headings = predictions.sum()
            high_confidence = sum((probabilities > 0.7) & (predictions == 1))
            
            logger.info(f"   📊 {predicted_headings} headings predicted ({high_confidence} high confidence)")
            logger.info(f"   🎯 Confidence range: {probabilities.min():.3f} - {probabilities.max():.3f}")
            
            return predictions_path
            
        except Exception as e:
            logger.error(f"❌ Prediction failed for {pdf_name}: {e}")
            return None
    
    def step3_wait_for_review(self, check_interval_minutes=5):
        """Step 3: Wait for user to review and correct predictions"""
        logger.info("\\n" + "="*70)
        logger.info("👁️  STEP 3: WAITING FOR YOUR REVIEW")
        logger.info("="*70)
        
        prediction_files = glob.glob(os.path.join(self.predictions_dir, "*_predictions.csv"))
        
        if not prediction_files:
            logger.warning("⚠️  No prediction files found to review")
            return True
        
        logger.info(f"📝 Found {len(prediction_files)} files for review")
        logger.info("\\n📋 REVIEW INSTRUCTIONS:")
        logger.info("=" * 30)
        logger.info("1. Open the CSV files in the 'predictions' folder")
        logger.info("2. Check the 'predicted_heading' column (1=heading, 0=not heading)")
        logger.info("3. Correct the 'is_heading' column where predictions are wrong:")
        logger.info("   - Set is_heading=1 for actual headings")
        logger.info("   - Set is_heading=0 for non-headings")
        logger.info("4. Optionally set user_reviewed=1 for rows you've checked")
        logger.info("5. Save the files when done")
        logger.info("\\n📍 Files to review:")
        for f in prediction_files:
            logger.info(f"   📄 {Path(f).name}")
        
        logger.info(f"\\n⏰ I'll check every {check_interval_minutes} minutes if you're done...")
        logger.info("💡 Type 'done' when you finish reviewing, or 'skip' to proceed")
        
        # Interactive wait
        while True:
            user_input = input(f"\\n[{datetime.now().strftime('%H:%M:%S')}] Status (done/skip/status): ").strip().lower()
            
            if user_input == 'done':
                logger.info("✅ User indicated review is complete")
                break
            elif user_input == 'skip':
                logger.info("⏭️  User chose to skip review")
                break
            elif user_input == 'status':
                self.show_review_status()
            else:
                logger.info("⏳ Continuing to wait... (type 'done', 'skip', or 'status')")
                time.sleep(check_interval_minutes * 60)
        
        return True
    
    def show_review_status(self):
        """Show current review status"""
        prediction_files = glob.glob(os.path.join(self.predictions_dir, "*_predictions.csv"))
        
        total_rows = 0
        reviewed_rows = 0
        corrected_headings = 0
        
        logger.info("\\n📊 REVIEW STATUS:")
        logger.info("-" * 40)
        
        for pred_file in prediction_files:
            try:
                df = pd.read_csv(pred_file)
                rows = len(df)
                reviewed = df['user_reviewed'].sum() if 'user_reviewed' in df.columns else 0
                headings = df['is_heading'].sum() if 'is_heading' in df.columns else 0
                
                total_rows += rows
                reviewed_rows += reviewed
                corrected_headings += headings
                
                filename = Path(pred_file).name
                logger.info(f"📄 {filename:<35}: {rows:>4} rows, {reviewed:>3} reviewed, {headings:>3} headings")
                
            except Exception as e:
                logger.info(f"❌ Error reading {Path(pred_file).name}: {e}")
        
        logger.info("-" * 40)
        logger.info(f"📊 Total: {total_rows} rows, {reviewed_rows} reviewed ({reviewed_rows/total_rows*100:.1f}%)")
        logger.info(f"📊 Corrected headings: {corrected_headings}")
    
    def step4_retrain_with_corrections(self):
        """Step 4: Retrain model using original data + user corrections"""
        logger.info("\\n" + "="*70)
        logger.info("🔄 STEP 4: RETRAINING WITH USER CORRECTIONS")
        logger.info("="*70)
        
        # Load original labeled data
        original_df = self.heading_pipeline.load_labeled_data(min_heading_percentage=1.0)
        if original_df is None:
            logger.error("❌ Cannot load original labeled data")
            return False
        
        logger.info(f"📊 Original data: {len(original_df)} rows, {original_df['is_heading'].sum()} headings")
        
        # Load corrected predictions
        prediction_files = glob.glob(os.path.join(self.predictions_dir, "*_predictions.csv"))
        corrected_data = []
        
        total_new_data = 0
        total_new_headings = 0
        
        for pred_file in prediction_files:
            try:
                df = pd.read_csv(pred_file)
                
                # Only use rows that have corrections (where is_heading was set)
                if 'is_heading' in df.columns:
                    # Remove rows where is_heading is still 0 and predicted_heading is also 0
                    # (these are likely uncorrected non-headings)
                    meaningful_data = df[
                        (df['is_heading'] == 1) |  # User marked as heading
                        (df['predicted_heading'] == 1) |  # Model predicted as heading
                        (df.get('user_reviewed', 0) == 1)  # User explicitly reviewed
                    ].copy()
                    
                    if len(meaningful_data) > 0:
                        # Add source info
                        meaningful_data['source_file'] = f"corrected_{Path(pred_file).stem}"
                        corrected_data.append(meaningful_data)
                        
                        headings_in_file = meaningful_data['is_heading'].sum()
                        total_new_data += len(meaningful_data)
                        total_new_headings += headings_in_file
                        
                        logger.info(f"✅ {Path(pred_file).name}: {len(meaningful_data)} meaningful rows, {headings_in_file} headings")
                
            except Exception as e:
                logger.error(f"❌ Error loading {Path(pred_file).name}: {e}")
        
        if not corrected_data:
            logger.warning("⚠️  No corrected data found. Training on original data only.")
            combined_df = original_df
        else:
            # Combine original + corrected data
            corrected_df = pd.concat(corrected_data, ignore_index=True)
            
            # Ensure same columns
            common_columns = set(original_df.columns) & set(corrected_df.columns)
            common_columns = list(common_columns)
            
            combined_df = pd.concat([
                original_df[common_columns],
                corrected_df[common_columns]
            ], ignore_index=True)
            
            logger.info(f"✅ Added {total_new_data} corrected rows with {total_new_headings} headings")
        
        logger.info(f"📊 Combined training data: {len(combined_df)} rows, {combined_df['is_heading'].sum()} headings ({combined_df['is_heading'].mean()*100:.1f}%)")
        
        # Retrain model
        self.current_cycle += 1
        logger.info(f"🔄 Starting training cycle {self.current_cycle}...")
        
        results = self.heading_pipeline.train_model(combined_df)
        
        # Save updated model
        model_path = self.heading_pipeline.save_model(f"cycle_{self.current_cycle}")
        
        logger.info("✅ Retraining completed!")
        logger.info(f"📈 New test accuracy: {results['test_accuracy']:.3f}")
        logger.info(f"🎯 New optimal threshold: {results['optimal_threshold']:.3f}")
        logger.info(f"💾 Updated model saved: {model_path}")
        
        return True
    
    def run_full_cycle(self, max_pdfs=None):
        """Run one complete cycle of the semi-supervised pipeline"""
        logger.info("\\n" + "="*80)
        logger.info("🔄 SEMI-SUPERVISED LEARNING PIPELINE - FULL CYCLE")
        logger.info("="*80)
        
        # Step 1: Initial training (only if no model exists)
        model_files = glob.glob(os.path.join(self.models_dir, 'working_heading_classifier_v*.pkl'))
        if not model_files:
            logger.info("🆕 No existing model found. Starting with initial training...")
            if not self.step1_initial_training():
                return False
        else:
            logger.info("✅ Using existing trained model")
            self.heading_pipeline.load_model()
        
        # Step 2: Process all PDFs
        if not self.step2_process_all_pdfs(max_pdfs):
            logger.error("❌ PDF processing failed")
            return False
        
        # Step 3: Wait for review
        if not self.step3_wait_for_review():
            return False
        
        # Step 4: Retrain with corrections
        if not self.step4_retrain_with_corrections():
            logger.error("❌ Retraining failed")
            return False
        
        logger.info("\\n🎉 FULL CYCLE COMPLETED SUCCESSFULLY!")
        logger.info("🔄 Ready for next cycle with improved model")
        
        return True
    
    def run_continuous_learning(self, max_cycles=5, max_pdfs_per_cycle=10):
        """Run multiple cycles for continuous learning"""
        logger.info("\\n" + "="*80)
        logger.info("🔄 CONTINUOUS SEMI-SUPERVISED LEARNING")
        logger.info("="*80)
        
        for cycle in range(max_cycles):
            logger.info(f"\\n🔄 Starting learning cycle {cycle + 1}/{max_cycles}")
            
            if not self.run_full_cycle(max_pdfs_per_cycle):
                logger.error(f"❌ Cycle {cycle + 1} failed")
                break
            
            # Ask user if they want to continue
            if cycle < max_cycles - 1:
                continue_learning = input("\\n🔄 Continue to next cycle? (y/N): ").strip().lower()
                if continue_learning != 'y':
                    logger.info("🛑 User chose to stop continuous learning")
                    break
        
        logger.info("\\n🏁 Continuous learning completed!")

def main():
    """Main function with menu"""
    print("🤖 SEMI-SUPERVISED HEADING DETECTION PIPELINE")
    print("=" * 50)
    print("This pipeline will:")
    print("1. Train model on your labeled data")
    print("2. Process ALL PDFs in unprocessed_pdfs/ folder")
    print("3. Generate predictions in predictions/ folder")
    print("4. Wait for your manual review/corrections")
    print("5. Retrain model with your corrections")
    print("6. Repeat for continuous improvement")
    print()
    
    pipeline = SemiSupervisedPipeline()
    
    while True:
        print("\\n📋 OPTIONS:")
        print("1. Run one full cycle")
        print("2. Run continuous learning (multiple cycles)")
        print("3. Process PDFs only (no retraining)")
        print("4. Show review status")
        print("5. Retrain with existing corrections")
        print("6. Exit")
        
        choice = input("\\nSelect option (1-6): ").strip()
        
        if choice == '1':
            max_pdfs = input("Max PDFs to process (Enter for all): ").strip()
            max_pdfs = int(max_pdfs) if max_pdfs.isdigit() else None
            pipeline.run_full_cycle(max_pdfs)
            
        elif choice == '2':
            max_cycles = input("Max cycles (default 5): ").strip()
            max_cycles = int(max_cycles) if max_cycles.isdigit() else 5
            max_pdfs = input("Max PDFs per cycle (default 10): ").strip()
            max_pdfs = int(max_pdfs) if max_pdfs.isdigit() else 10
            pipeline.run_continuous_learning(max_cycles, max_pdfs)
            
        elif choice == '3':
            max_pdfs = input("Max PDFs to process (Enter for all): ").strip()
            max_pdfs = int(max_pdfs) if max_pdfs.isdigit() else None
            if pipeline.heading_pipeline.load_model():
                pipeline.step2_process_all_pdfs(max_pdfs)
            else:
                print("❌ No trained model found. Run option 1 first.")
            
        elif choice == '4':
            pipeline.show_review_status()
            
        elif choice == '5':
            pipeline.step4_retrain_with_corrections()
            
        elif choice == '6':
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please select 1-6.")

if __name__ == "__main__":
    main()
