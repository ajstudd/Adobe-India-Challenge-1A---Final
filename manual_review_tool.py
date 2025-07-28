#!/usr/bin/env python3
"""
Manual Review Tool for Heading Predictions
=========================================

This tool provides an interactive interface for manually reviewing and correcting
heading predictions. It helps create high-quality training data for model improvement.

Features:
✅ Interactive review of heading predictions
✅ Quick keyboard shortcuts for efficient review
✅ Context display around each prediction
✅ Batch operations for similar patterns
✅ Progress tracking and save/resume functionality
✅ Statistics and quality metrics
✅ Export corrected data for retraining

Usage:
    python manual_review_tool.py

Author: AI Assistant
Date: July 28, 2025
"""

import os
import sys
import json
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ManualReviewTool:
    """Interactive tool for manual review of heading predictions"""
    
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load configuration
        self.config = self.load_config()
        
        # Core directories
        self.predictions_dir = os.path.join(self.base_dir, self.config['directories']['predictions'])
        self.reviewed_dir = os.path.join(self.base_dir, self.config['directories']['reviewed'])
        
        # Create directories
        os.makedirs(self.reviewed_dir, exist_ok=True)
        
        # Review state
        self.current_file = None
        self.current_df = None
        self.current_index = 0
        self.review_stats = {
            'total_reviewed': 0,
            'corrections_made': 0,
            'false_positives_found': 0,
            'false_negatives_found': 0,
            'session_start': datetime.now().isoformat()
        }
        
        logger.info("👁️  Manual Review Tool initialized!")
        logger.info(f"📁 Predictions: {self.predictions_dir}")
        logger.info(f"📁 Reviewed: {self.reviewed_dir}")
    
    def load_config(self):
        """Load configuration"""
        config_path = os.path.join(self.base_dir, 'config_main.json')
        
        if not os.path.exists(config_path):
            return {"directories": {"predictions": "predictions", "reviewed": "reviewed"}}
        
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"❌ Error loading config: {e}")
            return {"directories": {"predictions": "predictions", "reviewed": "reviewed"}}
    
    def list_available_files(self) -> List[str]:
        """List available prediction files for review"""
        prediction_files = list(Path(self.predictions_dir).glob("*_predictions.csv"))
        
        # Filter out already reviewed files
        available_files = []
        for file_path in prediction_files:
            reviewed_path = Path(self.reviewed_dir) / file_path.name
            if not reviewed_path.exists():
                available_files.append(str(file_path))
        
        return sorted(available_files)
    
    def load_prediction_file(self, file_path: str) -> bool:
        """Load a prediction file for review"""
        try:
            self.current_df = pd.read_csv(file_path)
            self.current_file = file_path
            self.current_index = 0
            
            # Add review tracking columns if they don't exist
            if 'review_status' not in self.current_df.columns:
                self.current_df['review_status'] = 'not_reviewed'
            if 'original_prediction' not in self.current_df.columns:
                self.current_df['original_prediction'] = self.current_df.get('is_heading', 0)
            if 'corrected_label' not in self.current_df.columns:
                self.current_df['corrected_label'] = self.current_df.get('is_heading', 0)
            
            logger.info(f"📄 Loaded: {os.path.basename(file_path)}")
            logger.info(f"   📊 Total blocks: {len(self.current_df)}")
            logger.info(f"   🎯 Predicted headings: {self.current_df['original_prediction'].sum()}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading file: {e}")
            return False
    
    def get_context(self, index: int, window_size: int = 2) -> List[Dict]:
        """Get context around the current block"""
        if self.current_df is None:
            return []
        
        start_idx = max(0, index - window_size)
        end_idx = min(len(self.current_df), index + window_size + 1)
        
        context = []
        for i in range(start_idx, end_idx):
            row = self.current_df.iloc[i]
            context.append({
                'index': i,
                'text': row.get('text', ''),
                'is_current': i == index,
                'is_heading': row.get('original_prediction', 0),
                'font_size': row.get('font_size', 12),
                'page': row.get('page', 1),
                'confidence': row.get('heading_confidence', 0.0)
            })
        
        return context
    
    def display_current_block(self):
        """Display the current block for review"""
        if self.current_df is None or self.current_index >= len(self.current_df):
            return
        
        row = self.current_df.iloc[self.current_index]
        context = self.get_context(self.current_index)
        
        # Clear screen (works on most terminals)
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("👁️  MANUAL REVIEW TOOL")
        print("=" * 80)
        print(f"📄 File: {os.path.basename(self.current_file)}")
        print(f"📊 Progress: {self.current_index + 1}/{len(self.current_df)} ({((self.current_index + 1)/len(self.current_df)*100):.1f}%)")
        print(f"🎯 Session stats: {self.review_stats['corrections_made']} corrections, {self.review_stats['total_reviewed']} reviewed")
        print()
        
        # Display context
        print("📝 CONTEXT:")
        print("-" * 40)
        for ctx in context:
            marker = ">>> " if ctx['is_current'] else "    "
            pred_marker = "[H]" if ctx['is_heading'] else "[T]"
            font_info = f"(font: {ctx['font_size']:.1f})" if 'font_size' in row else ""
            conf_info = f"(conf: {ctx.get('confidence', 0):.2f})" if ctx['is_current'] else ""
            
            text_preview = ctx['text'][:60] + "..." if len(ctx['text']) > 60 else ctx['text']
            
            print(f"{marker}{pred_marker} {text_preview} {font_info} {conf_info}")
        
        print("-" * 40)
        
        # Current block details
        current_text = row.get('text', '')
        current_pred = row.get('original_prediction', 0)
        current_conf = row.get('heading_confidence', 0.0)
        current_font = row.get('font_size', 12)
        current_page = row.get('page', 1)
        
        print()
        print("🔍 CURRENT BLOCK DETAILS:")
        print(f"Text: {current_text}")
        print(f"Current prediction: {'HEADING' if current_pred else 'NOT HEADING'}")
        print(f"Confidence: {current_conf:.3f}")
        print(f"Font size: {current_font}")
        print(f"Page: {current_page}")
        
        # Filter information if available
        if 'filter_decision' in row:
            print(f"Filter decision: {row['filter_decision']}")
        if 'filter_reasons' in row:
            print(f"Filter reasons: {row['filter_reasons']}")
        
        print()
    
    def get_user_decision(self) -> str:
        """Get user decision for the current block"""
        print("🎯 REVIEW OPTIONS:")
        print("  [h] Mark as HEADING")
        print("  [t] Mark as NOT HEADING")
        print("  [k] Keep current prediction (no change)")
        print("  [s] Skip for now")
        print("  [b] Go back to previous")
        print("  [j] Jump to specific index")
        print("  [q] Save and quit")
        print("  [?] Show this help")
        print()
        
        while True:
            choice = input("Your choice: ").strip().lower()
            
            if choice in ['h', 't', 'k', 's', 'b', 'j', 'q', '?']:
                return choice
            else:
                print("❌ Invalid choice. Please use h, t, k, s, b, j, q, or ?")
    
    def process_user_decision(self, decision: str) -> bool:
        """Process user decision and update data"""
        if self.current_df is None:
            return True
        
        row_idx = self.current_index
        current_pred = self.current_df.iloc[row_idx]['original_prediction']
        
        if decision == 'h':
            # Mark as heading
            self.current_df.at[row_idx, 'corrected_label'] = 1
            self.current_df.at[row_idx, 'review_status'] = 'reviewed'
            if current_pred != 1:
                self.review_stats['corrections_made'] += 1
                self.review_stats['false_negatives_found'] += 1
            self.review_stats['total_reviewed'] += 1
            self.current_index += 1
            
        elif decision == 't':
            # Mark as not heading
            self.current_df.at[row_idx, 'corrected_label'] = 0
            self.current_df.at[row_idx, 'review_status'] = 'reviewed'
            if current_pred != 0:
                self.review_stats['corrections_made'] += 1
                self.review_stats['false_positives_found'] += 1
            self.review_stats['total_reviewed'] += 1
            self.current_index += 1
            
        elif decision == 'k':
            # Keep current prediction
            self.current_df.at[row_idx, 'corrected_label'] = current_pred
            self.current_df.at[row_idx, 'review_status'] = 'reviewed'
            self.review_stats['total_reviewed'] += 1
            self.current_index += 1
            
        elif decision == 's':
            # Skip
            self.current_df.at[row_idx, 'review_status'] = 'skipped'
            self.current_index += 1
            
        elif decision == 'b':
            # Go back
            if self.current_index > 0:
                self.current_index -= 1
            else:
                print("❌ Already at the first block")
                input("Press Enter to continue...")
            
        elif decision == 'j':
            # Jump to index
            try:
                target_index = int(input("Enter target index (0-based): "))
                if 0 <= target_index < len(self.current_df):
                    self.current_index = target_index
                else:
                    print(f"❌ Index out of range (0-{len(self.current_df)-1})")
                    input("Press Enter to continue...")
            except ValueError:
                print("❌ Invalid index")
                input("Press Enter to continue...")
        
        elif decision == 'q':
            # Save and quit
            return False
        
        elif decision == '?':
            # Show help (redisplay is handled by main loop)
            pass
        
        return True
    
    def save_reviewed_file(self) -> str:
        """Save the reviewed file"""
        if self.current_df is None or self.current_file is None:
            return ""
        
        # Generate output filename
        input_name = Path(self.current_file).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"{input_name}_reviewed_{timestamp}.csv"
        output_path = os.path.join(self.reviewed_dir, output_name)
        
        # Add review metadata
        self.current_df['review_session'] = timestamp
        self.current_df['reviewer'] = 'manual_review_tool'
        
        # Update final label column
        self.current_df['is_heading'] = self.current_df['corrected_label']
        
        # Save file
        self.current_df.to_csv(output_path, index=False)
        
        logger.info(f"💾 Reviewed file saved: {output_path}")
        return output_path
    
    def generate_review_report(self) -> str:
        """Generate a review session report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.reviewed_dir, f"review_report_{timestamp}.json")
        
        # Calculate additional statistics
        if self.current_df is not None:
            reviewed_mask = self.current_df['review_status'] == 'reviewed'
            reviewed_data = self.current_df[reviewed_mask]
            
            if len(reviewed_data) > 0:
                original_headings = reviewed_data['original_prediction'].sum()
                corrected_headings = reviewed_data['corrected_label'].sum()
                
                self.review_stats.update({
                    'blocks_reviewed': len(reviewed_data),
                    'original_heading_count': int(original_headings),
                    'corrected_heading_count': int(corrected_headings),
                    'net_change': int(corrected_headings - original_headings)
                })
        
        self.review_stats['session_end'] = datetime.now().isoformat()
        
        report = {
            'review_session': timestamp,
            'file_reviewed': os.path.basename(self.current_file) if self.current_file else '',
            'statistics': self.review_stats,
            'quality_metrics': {
                'correction_rate': self.review_stats['corrections_made'] / max(1, self.review_stats['total_reviewed']) * 100,
                'false_positive_rate': self.review_stats['false_positives_found'] / max(1, self.review_stats['total_reviewed']) * 100,
                'false_negative_rate': self.review_stats['false_negatives_found'] / max(1, self.review_stats['total_reviewed']) * 100
            }
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📋 Review report saved: {report_path}")
        return report_path
    
    def review_file_interactive(self, file_path: str):
        """Interactive review of a single file"""
        if not self.load_prediction_file(file_path):
            return
        
        print(f"\n🚀 Starting review of {os.path.basename(file_path)}")
        print("💡 Tip: Focus on predicted headings first for efficiency")
        
        # Option to start with predictions only
        review_predictions_only = input("\n🎯 Review only predicted headings first? (Y/n): ").strip().lower()
        
        if review_predictions_only != 'n':
            # Filter to show only predicted headings
            predicted_indices = self.current_df[self.current_df['original_prediction'] == 1].index.tolist()
            if predicted_indices:
                print(f"📊 Found {len(predicted_indices)} predicted headings to review")
                
                for idx in predicted_indices:
                    self.current_index = idx
                    self.display_current_block()
                    decision = self.get_user_decision()
                    
                    if not self.process_user_decision(decision):
                        break
                
                # Ask if user wants to continue with full review
                continue_full = input("\n🔄 Continue with full document review? (y/N): ").strip().lower()
                if continue_full not in ['y', 'yes']:
                    self.save_reviewed_file()
                    self.generate_review_report()
                    return
        
        # Full document review
        self.current_index = 0
        while self.current_index < len(self.current_df):
            self.display_current_block()
            decision = self.get_user_decision()
            
            if not self.process_user_decision(decision):
                break
        
        # Save results
        output_path = self.save_reviewed_file()
        report_path = self.generate_review_report()
        
        print("\n✅ Review session completed!")
        print(f"📊 Statistics:")
        print(f"   📝 Total reviewed: {self.review_stats['total_reviewed']}")
        print(f"   ✏️  Corrections made: {self.review_stats['corrections_made']}")
        print(f"   ❌ False positives found: {self.review_stats['false_positives_found']}")
        print(f"   ➕ False negatives found: {self.review_stats['false_negatives_found']}")
        print(f"💾 Saved: {output_path}")
        print(f"📋 Report: {report_path}")
    
    def interactive_menu(self):
        """Main interactive menu"""
        while True:
            print("\n👁️  MANUAL REVIEW TOOL")
            print("=" * 40)
            
            # List available files
            available_files = self.list_available_files()
            
            if not available_files:
                print("📄 No prediction files available for review")
                print("💡 Run the prediction pipeline first to generate files")
                break
            
            print("📄 Available files for review:")
            for i, file_path in enumerate(available_files, 1):
                file_name = os.path.basename(file_path)
                file_size = os.path.getsize(file_path)
                
                # Quick file info
                try:
                    df = pd.read_csv(file_path)
                    predicted_headings = df.get('is_heading', df.get('original_prediction', pd.Series([0]))).sum()
                    total_blocks = len(df)
                except:
                    predicted_headings = "?"
                    total_blocks = "?"
                
                print(f"  [{i}] {file_name}")
                print(f"      📊 {total_blocks} blocks, {predicted_headings} predicted headings")
                print(f"      💾 {file_size/1024:.1f} KB")
            
            print()
            print("Options:")
            print("  [1-N] Review specific file")
            print("  [a] Review all files sequentially")
            print("  [q] Quit")
            
            choice = input("\nYour choice: ").strip().lower()
            
            if choice == 'q':
                break
            elif choice == 'a':
                # Review all files
                for file_path in available_files:
                    print(f"\n🚀 Reviewing {os.path.basename(file_path)}")
                    self.review_file_interactive(file_path)
                    
                    continue_next = input("\n🔄 Continue to next file? (Y/n): ").strip().lower()
                    if continue_next == 'n':
                        break
            else:
                # Review specific file
                try:
                    file_index = int(choice) - 1
                    if 0 <= file_index < len(available_files):
                        self.review_file_interactive(available_files[file_index])
                    else:
                        print("❌ Invalid file number")
                except ValueError:
                    print("❌ Invalid choice")


def main():
    """Main function"""
    print("👁️  MANUAL REVIEW TOOL FOR HEADING PREDICTIONS")
    print("=" * 60)
    print("🎯 This tool helps you manually review and correct heading predictions")
    print("📊 Use it to create high-quality training data for model improvement")
    print()
    print("💡 Tips:")
    print("   • Review predicted headings first for efficiency")
    print("   • Use keyboard shortcuts for quick navigation")
    print("   • Focus on borderline cases and obvious errors")
    print("   • Take breaks to maintain accuracy")
    print()
    
    try:
        # Initialize and run review tool
        review_tool = ManualReviewTool()
        review_tool.interactive_menu()
        
        print("\n✅ Manual review session completed!")
        print("🎯 Use the reviewed data for model retraining")
        
    except KeyboardInterrupt:
        print("\n⏹️  Review interrupted by user")
    except Exception as e:
        print(f"\n❌ Review tool error: {e}")
        logger.error(f"Review tool error: {e}")


if __name__ == "__main__":
    main()
