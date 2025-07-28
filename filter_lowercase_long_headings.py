#!/usr/bin/env python3
"""
Filter Lowercase Long Headings
=============================

A simple script to discard headings that start with lowercase letters and are long.

This script applies a simple rule:
- If a heading starts with a lowercase letter AND is longer than a threshold, discard it
- Default threshold is 50 characters, but can be customized

Usage:
    python filter_lowercase_long_headings.py

Author: AI Assistant
Date: July 28, 2025
"""

import pandas as pd
import os
import json
import logging
from typing import List, Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LowercaseLongHeadingFilter:
    """Simple filter to remove headings that start with lowercase and are long"""
    
    def __init__(self, max_length_threshold: int = 50):
        """
        Initialize the filter
        
        Args:
            max_length_threshold: Maximum length for headings starting with lowercase
        """
        self.max_length_threshold = max_length_threshold
        logger.info(f"🔧 Lowercase long heading filter initialized")
        logger.info(f"📏 Max length threshold: {max_length_threshold} characters")
    
    def should_discard_heading(self, text: str) -> bool:
        """
        Check if a heading should be discarded based on lowercase start + length
        
        Args:
            text: The heading text to check
            
        Returns:
            True if the heading should be discarded, False otherwise
        """
        if not text or not isinstance(text, str):
            return False
            
        text_stripped = text.strip()
        
        # Check if starts with lowercase letter
        if not text_stripped or not text_stripped[0].islower():
            return False
            
        # Check if it's longer than threshold
        if len(text_stripped) > self.max_length_threshold:
            logger.debug(f"🗑️ Discarding long lowercase heading: '{text_stripped[:30]}...' (length: {len(text_stripped)})")
            return True
            
        return False
    
    def filter_dataframe(self, df: pd.DataFrame, text_column: str = 'text', 
                        heading_column: str = 'is_heading') -> pd.DataFrame:
        """
        Filter a DataFrame to remove long lowercase headings
        
        Args:
            df: DataFrame containing the data
            text_column: Name of the column containing text
            heading_column: Name of the column indicating if it's a heading
            
        Returns:
            Filtered DataFrame
        """
        if text_column not in df.columns:
            logger.error(f"❌ Column '{text_column}' not found in DataFrame")
            return df
            
        if heading_column not in df.columns:
            logger.error(f"❌ Column '{heading_column}' not found in DataFrame")
            return df
        
        filtered_df = df.copy()
        initial_heading_count = filtered_df[heading_column].sum()
        
        # Apply filter only to rows marked as headings
        heading_mask = filtered_df[heading_column] == 1
        
        for idx in filtered_df[heading_mask].index:
            text = filtered_df.at[idx, text_column]
            if self.should_discard_heading(text):
                filtered_df.at[idx, heading_column] = 0
                logger.info(f"🗑️ Discarded: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        
        final_heading_count = filtered_df[heading_column].sum()
        removed_count = initial_heading_count - final_heading_count
        
        logger.info(f"📊 Filter results:")
        logger.info(f"   Initial headings: {initial_heading_count}")
        logger.info(f"   Final headings: {final_heading_count}")
        logger.info(f"   Removed: {removed_count}")
        
        return filtered_df
    
    def filter_json_headings(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter headings in JSON format output
        
        Args:
            json_data: JSON data containing headings
            
        Returns:
            Filtered JSON data
        """
        filtered_data = json_data.copy()
        
        if 'headings' not in filtered_data:
            logger.warning("⚠️ No 'headings' key found in JSON data")
            return filtered_data
        
        original_headings = filtered_data['headings']
        filtered_headings = []
        removed_count = 0
        
        for heading in original_headings:
            if isinstance(heading, dict) and 'text' in heading:
                text = heading['text']
                if self.should_discard_heading(text):
                    removed_count += 1
                    logger.info(f"🗑️ Discarded JSON heading: '{text[:50]}{'...' if len(text) > 50 else ''}'")
                else:
                    filtered_headings.append(heading)
            else:
                # Keep non-dict items or items without text
                filtered_headings.append(heading)
        
        filtered_data['headings'] = filtered_headings
        
        logger.info(f"📊 JSON Filter results:")
        logger.info(f"   Original headings: {len(original_headings)}")
        logger.info(f"   Filtered headings: {len(filtered_headings)}")
        logger.info(f"   Removed: {removed_count}")
        
        return filtered_data

def process_csv_file(file_path: str, output_path: str = None, max_length: int = 50):
    """
    Process a CSV file to filter out long lowercase headings
    
    Args:
        file_path: Path to the CSV file
        output_path: Output path (if None, overwrites original)
        max_length: Maximum length threshold for lowercase headings
    """
    if not os.path.exists(file_path):
        logger.error(f"❌ File not found: {file_path}")
        return
    
    logger.info(f"📄 Processing CSV file: {os.path.basename(file_path)}")
    
    try:
        df = pd.read_csv(file_path)
        
        # Initialize filter
        filter_obj = LowercaseLongHeadingFilter(max_length)
        
        # Apply filter
        filtered_df = filter_obj.filter_dataframe(df)
        
        # Save result
        if output_path is None:
            output_path = file_path
        
        filtered_df.to_csv(output_path, index=False)
        logger.info(f"✅ Filtered CSV saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"❌ Error processing CSV file: {e}")

def process_json_file(file_path: str, output_path: str = None, max_length: int = 50):
    """
    Process a JSON file to filter out long lowercase headings
    
    Args:
        file_path: Path to the JSON file
        output_path: Output path (if None, overwrites original)
        max_length: Maximum length threshold for lowercase headings
    """
    if not os.path.exists(file_path):
        logger.error(f"❌ File not found: {file_path}")
        return
    
    logger.info(f"📄 Processing JSON file: {os.path.basename(file_path)}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        # Initialize filter
        filter_obj = LowercaseLongHeadingFilter(max_length)
        
        # Apply filter
        filtered_data = filter_obj.filter_json_headings(json_data)
        
        # Save result
        if output_path is None:
            output_path = file_path
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Filtered JSON saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"❌ Error processing JSON file: {e}")

def main():
    """Main function for interactive usage"""
    print("🗑️ LOWERCASE LONG HEADING FILTER")
    print("=" * 50)
    print("📋 This script removes headings that:")
    print("   • Start with a lowercase letter")
    print("   • Are longer than the specified threshold")
    print()
    
    while True:
        print("Options:")
        print("1. Filter CSV file")
        print("2. Filter JSON file")
        print("3. Process all CSV files in output folder")
        print("4. Process all JSON files in output folder")
        print("5. Exit")
        
        choice = input("\nChoose an option (1-5): ").strip()
        
        if choice == '1':
            file_path = input("Enter CSV file path: ").strip().strip('"')
            max_length = input("Enter max length threshold (default 50): ").strip()
            max_length = int(max_length) if max_length.isdigit() else 50
            process_csv_file(file_path, max_length=max_length)
            
        elif choice == '2':
            file_path = input("Enter JSON file path: ").strip().strip('"')
            max_length = input("Enter max length threshold (default 50): ").strip()
            max_length = int(max_length) if max_length.isdigit() else 50
            process_json_file(file_path, max_length=max_length)
            
        elif choice == '3':
            output_dir = 'output'
            max_length = input("Enter max length threshold (default 50): ").strip()
            max_length = int(max_length) if max_length.isdigit() else 50
            
            if os.path.exists(output_dir):
                csv_files = [f for f in os.listdir(output_dir) if f.endswith('.csv')]
                for csv_file in csv_files:
                    process_csv_file(os.path.join(output_dir, csv_file), max_length=max_length)
            else:
                logger.error(f"❌ Output directory not found: {output_dir}")
                
        elif choice == '4':
            output_dir = 'output'
            max_length = input("Enter max length threshold (default 50): ").strip()
            max_length = int(max_length) if max_length.isdigit() else 50
            
            if os.path.exists(output_dir):
                json_files = [f for f in os.listdir(output_dir) if f.endswith('.json')]
                for json_file in json_files:
                    process_json_file(os.path.join(output_dir, json_file), max_length=max_length)
            else:
                logger.error(f"❌ Output directory not found: {output_dir}")
                
        elif choice == '5':
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please try again.")
        
        print()

if __name__ == "__main__":
    main()
