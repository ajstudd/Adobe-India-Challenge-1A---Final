"""
PDF Extraction Integration for Clean Pipeline
============================================

This module integrates with your existing PDF extraction capabilities
and prepares data for the heading detection pipeline.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logger = logging.getLogger(__name__)

# Try to import your existing PDF processing scripts
try:
    from scripts.predict_outline_from_pdfs import extract_pdf_blocks
    PDF_EXTRACTION_METHOD_1 = True
except (ImportError, FileNotFoundError, Exception) as e:
    print(f"PDF extraction method 1 not available: {e}")
    PDF_EXTRACTION_METHOD_1 = False

try:
    from src.extraction.extract_local_dataset_to_csv import process_pdf
    PDF_EXTRACTION_METHOD_2 = True
except (ImportError, FileNotFoundError, Exception) as e:
    print(f"PDF extraction method 2 not available: {e}")
    PDF_EXTRACTION_METHOD_2 = False

class PDFExtractor:
    """Handles PDF extraction for the clean pipeline"""
    
    def __init__(self):
        self.base_dir = os.path.dirname(__file__)
        self.project_root = os.path.abspath(os.path.join(self.base_dir, '..'))
        
        # Check available extraction methods
        self.available_methods = []
        if PDF_EXTRACTION_METHOD_1:
            self.available_methods.append("predict_outline_from_pdfs")
        if PDF_EXTRACTION_METHOD_2:
            self.available_methods.append("extract_local_dataset_to_csv")
        
        logger.info(f"📄 Available PDF extraction methods: {self.available_methods}")
    
    def extract_pdf_to_blocks(self, pdf_path, output_csv=None):
        """
        Extract PDF to blocks using available methods
        Returns DataFrame with all required features
        """
        pdf_name = Path(pdf_path).stem
        
        if output_csv is None:
            output_csv = os.path.join(self.base_dir, 'extracted_csvs', f"{pdf_name}_blocks.csv")
        
        # Create output directory
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        
        # Try extraction methods in order of preference
        df = None
        
        if PDF_EXTRACTION_METHOD_2:
            try:
                logger.info(f"🔧 Trying extract_local_dataset_to_csv for {pdf_name}...")
                result = process_pdf(pdf_path, output_csv)
                
                if os.path.exists(output_csv):
                    df = pd.read_csv(output_csv)
                    logger.info(f"✅ Extracted {len(df)} blocks using extract_local_dataset_to_csv")
                
            except Exception as e:
                logger.warning(f"⚠️  extract_local_dataset_to_csv failed: {e}")
        
        if df is None and PDF_EXTRACTION_METHOD_1:
            try:
                logger.info(f"🔧 Trying predict_outline_from_pdfs for {pdf_name}...")
                df = extract_pdf_blocks(pdf_path)
                
                if df is not None and len(df) > 0:
                    logger.info(f"✅ Extracted {len(df)} blocks using predict_outline_from_pdfs")
                
            except Exception as e:
                logger.warning(f"⚠️  predict_outline_from_pdfs failed: {e}")
        
        # Always use fallback extraction for now since the dependencies have issues
        if df is None:
            logger.info(f"📄 Using PyMuPDF fallback extraction for {pdf_name}...")
            df = self.create_fallback_extraction(pdf_path)
        
        if df is None or len(df) == 0:
            logger.error(f"❌ No blocks extracted from {pdf_name}")
            return None
        
        # Ensure all required columns exist
        df = self.standardize_columns(df, pdf_name)
        
        # Save to CSV
        df.to_csv(output_csv, index=False)
        logger.info(f"💾 Saved blocks to: {output_csv}")
        
        return df
    
    def create_fallback_extraction(self, pdf_path):
        """Create a basic fallback extraction using PyMuPDF directly"""
        try:
            import fitz  # PyMuPDF
            
            pdf_name = Path(pdf_path).stem
            doc = fitz.open(pdf_path)
            
            blocks = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text_dict = page.get_text("dict")
                
                for block in text_dict["blocks"]:
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                text = span["text"].strip()
                                if len(text) > 0:
                                    blocks.append({
                                        'text': text,
                                        'font_size': span.get("size", 12),
                                        'page': page_num + 1,
                                        'x0': span.get("bbox", [0, 0, 0, 0])[0],
                                        'y0': span.get("bbox", [0, 0, 0, 0])[1],
                                        'x1': span.get("bbox", [0, 0, 0, 0])[2],
                                        'y1': span.get("bbox", [0, 0, 0, 0])[3],
                                        'font': span.get("font", "unknown"),
                                        'bold': 'Bold' in span.get("flags", 0),
                                        'italic': 'Italic' in span.get("flags", 0)
                                    })
            
            doc.close()
            
            if blocks:
                df = pd.DataFrame(blocks)
                logger.info(f"✅ Fallback extraction: {len(df)} blocks from {pdf_name}")
                return df
            else:
                logger.error(f"❌ No text blocks extracted from {pdf_name}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Fallback extraction failed: {e}")
            return None
    
    def standardize_columns(self, df, source_name):
        """Ensure all required columns exist with default values"""
        
        # Required columns with default values
        required_columns = {
            'text': '',
            'font_size': 12.0,
            'page': 1,
            'x0': 0.0,
            'y0': 0.0,
            'x1': 0.0,
            'y1': 0.0,
            'font': 'unknown',
            'bold': False,
            'italic': False,
            'underline': False,
            'color': 0,
            'bullet': False,
            'math': False,
            'hyperlink': False,
            'is_all_caps': False,
            'is_title_case': False,
            'ends_with_colon': False,
            'starts_with_number': False,
            'punctuation_count': 0,
            'contains_colon': False,
            'contains_semicolon': False,
            'word_count': 0,
            'line_position_on_page': 0.0,
            'relative_font_size': 1.0,
            'distance_to_previous_heading': 0.0,
            'line_spacing_above': 0.0,
            'num_nouns': 0,
            'num_verbs': 0,
            'num_adjs': 0,
            'num_advs': 0,
            'num_propn': 0,
            'num_pronouns': 0,
            'num_other_pos': 0,
            'source_file': source_name
        }
        
        # Add missing columns
        for col, default_val in required_columns.items():
            if col not in df.columns:
                df[col] = default_val
        
        # Calculate derived features
        df = self.calculate_derived_features(df)
        
        return df
    
    def calculate_derived_features(self, df):
        """Calculate derived features that might be missing"""
        
        # Text-based features
        df['text'] = df['text'].fillna('').astype(str)
        df['word_count'] = df['text'].str.split().str.len()
        df['is_all_caps'] = df['text'].str.isupper().astype(int)
        df['is_title_case'] = df['text'].str.istitle().astype(int)
        df['ends_with_colon'] = df['text'].str.endswith(':').astype(int)
        df['starts_with_number'] = df['text'].str.match(r'^\d').fillna(False).astype(int)
        df['contains_colon'] = df['text'].str.contains(':').astype(int)
        df['contains_semicolon'] = df['text'].str.contains(';').astype(int)
        
        # Punctuation count
        import string
        df['punctuation_count'] = df['text'].apply(
            lambda x: sum(1 for c in x if c in string.punctuation)
        )
        
        # Font size features
        if 'font_size' in df.columns:
            df['font_size'] = pd.to_numeric(df['font_size'], errors='coerce').fillna(12)
            mean_font_size = df['font_size'].mean()
            df['relative_font_size'] = df['font_size'] / mean_font_size
        
        # Position features
        numeric_cols = ['x0', 'y0', 'x1', 'y1', 'page']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Page position
        if 'y0' in df.columns and 'page' in df.columns:
            df['line_position_on_page'] = df.groupby('page')['y0'].rank(ascending=False, method='dense')
            df['line_position_on_page'] = df['line_position_on_page'] / df.groupby('page')['y0'].transform('count')
        
        # Line spacing (simplified)
        if 'y0' in df.columns:
            df = df.sort_values(['page', 'y0'], ascending=[True, False])
            df['line_spacing_above'] = df.groupby('page')['y0'].diff().abs().fillna(0)
        
        # Boolean conversions
        bool_cols = ['bold', 'italic', 'underline', 'bullet', 'math', 'hyperlink']
        for col in bool_cols:
            if col in df.columns:
                df[col] = df[col].fillna(False).astype(int)
        
        return df
    
    def process_batch_pdfs(self, pdf_dir, output_dir, max_files=10):
        """Process a batch of PDFs"""
        import glob
        
        pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
        
        if not pdf_files:
            logger.warning(f"⚠️  No PDF files found in {pdf_dir}")
            return []
        
        pdf_files = pdf_files[:max_files]
        logger.info(f"📄 Processing {len(pdf_files)} PDFs...")
        
        results = []
        
        for i, pdf_path in enumerate(pdf_files):
            pdf_name = Path(pdf_path).stem
            logger.info(f"\n--- PDF {i+1}/{len(pdf_files)}: {pdf_name} ---")
            
            try:
                output_csv = os.path.join(output_dir, f"{pdf_name}_blocks.csv")
                df = self.extract_pdf_to_blocks(pdf_path, output_csv)
                
                if df is not None:
                    results.append({
                        'pdf_name': pdf_name,
                        'pdf_path': pdf_path,
                        'csv_path': output_csv,
                        'blocks_count': len(df),
                        'status': 'success'
                    })
                else:
                    results.append({
                        'pdf_name': pdf_name,
                        'pdf_path': pdf_path,
                        'status': 'failed'
                    })
                    
            except Exception as e:
                logger.error(f"❌ Error processing {pdf_name}: {e}")
                results.append({
                    'pdf_name': pdf_name,
                    'pdf_path': pdf_path,
                    'status': 'error',
                    'error': str(e)
                })
        
        successful = [r for r in results if r['status'] == 'success']
        logger.info(f"✅ Successfully processed {len(successful)}/{len(pdf_files)} PDFs")
        
        return results

def test_extractor():
    """Test the PDF extractor"""
    extractor = PDFExtractor()
    
    # Test with one PDF
    pdf_dir = os.path.join('..', 'unprocessed_pdfs')
    output_dir = os.path.join('.', 'test_extractions')
    
    results = extractor.process_batch_pdfs(pdf_dir, output_dir, max_files=1)
    
    if results:
        print(f"✅ Test completed: {results[0]}")
    else:
        print("❌ Test failed")

if __name__ == "__main__":
    test_extractor()
