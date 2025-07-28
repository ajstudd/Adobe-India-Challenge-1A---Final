import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import json
from src.utils.pdf_utils import extract_pdf_text_with_fonts
from src.utils.text_utils import clean_text
import fitz

# Paths
MODEL_PATH = os.path.join('models', 'docbank_heading_classifier_local.pkl')
INPUT_DIR = 'input'
OUTPUT_DIR = os.path.join('output', 'model_outputs')

# Features used for prediction (update as per your model)

# Features used for prediction (update as per your model)
FEATURES = [
    'x0', 'y0', 'x1', 'y1', 'page', 'font_size', 'font', 'color',
    'bold', 'italic', 'underline', 'bullet', 'math', 'hyperlink',
    'is_all_caps', 'is_title_case', 'ends_with_colon', 'starts_with_number',
    'punctuation_count', 'contains_colon', 'contains_semicolon', 'word_count',
    'line_position_on_page', 'relative_font_size', 'distance_to_previous_heading', 'line_spacing_above',
    'text_length', 'contains_keyword'
]


# Load trained model, TF-IDF vectorizer, and selected features
clf = joblib.load(MODEL_PATH)
TFIDF_VECTORIZER_PATH = os.path.join('models', 'docbank_heading_classifier_local_tfidf_vectorizer.pkl')
tfidf_vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
SELECTED_FEATURES_PATH = os.path.join('models', 'docbank_heading_classifier_local_selected_features.pkl')
selected_features = joblib.load(SELECTED_FEATURES_PATH)

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Process each PDF in input folder
for fname in os.listdir(INPUT_DIR):
    if not fname.lower().endswith('.pdf'):
        continue
    pdf_path = os.path.join(INPUT_DIR, fname)
    pdf_name = os.path.splitext(fname)[0]
    doc = fitz.open(pdf_path)
    all_blocks = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = extract_pdf_text_with_fonts(page)
        for block in blocks:
            block['page'] = page_num + 1  # 1-based page number
            all_blocks.append(block)
    if not all_blocks:
        print(f"No blocks found in {fname}, skipping.")
        continue
    # Convert to DataFrame and fill missing features
    df = pd.DataFrame(all_blocks)
    # Clean text
    df['text'] = df['text'].astype(str).apply(clean_text)
    
    # EXACTLY match training preprocessing
    # Encode categorical features FIRST
    from sklearn.preprocessing import LabelEncoder
    for col in ['font', 'color']:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna('None')
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
    
    # Convert boolean features to int (match training)
    for col in ['bold', 'italic', 'underline', 'bullet', 'math', 'hyperlink',
                'is_all_caps', 'is_title_case', 'ends_with_colon', 'starts_with_number']:
        if col in df.columns:
            df[col] = df[col].astype(float).fillna(0).astype(int)
    
    # Fill missing numeric features with 0 (match training)
    for col in ['punctuation_count', 'line_position_on_page', 'relative_font_size',
                'distance_to_previous_heading', 'line_spacing_above',
                'contains_colon', 'contains_semicolon', 'word_count']:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
    # Fill missing POS features with 0 if present
    pos_cols = ['num_nouns', 'num_verbs', 'num_adjs', 'num_advs', 'num_propn', 'num_pronouns', 'num_other_pos']
    for col in pos_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        else:
            df[col] = 0
    
    # Add text-based features (match training exactly)
    df['text_length'] = df['text'].apply(len)
    heading_keywords = [
        'introduction', 'background', 'abstract', 'conclusion', 'summary',
        'references', 'related work', 'discussion', 'results', 'methods',
        'chapter', 'section', 'contents', 'table of contents', 'appendix',
        'overview', 'analysis', 'scope', 'objective', 'aim', 'purpose'
    ]
    def contains_heading_keyword(text):
        text_lower = text.lower()
        return int(any(kw in text_lower for kw in heading_keywords))
    df['contains_keyword'] = df['text'].apply(contains_heading_keyword)
    
    # Fill missing basic features
    for col in FEATURES:
        if col not in df.columns:
            df[col] = 0
    # TF-IDF features for inference
    tfidf_matrix = tfidf_vectorizer.transform(df['text'])
    tfidf_feature_names = [f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])]
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_feature_names, index=df.index)
    
    # Concatenate TF-IDF features to X
    X_full = pd.concat([df[FEATURES], tfidf_df], axis=1)
    
    # Add dynamic font thresholding features (EXACT match to training)
    df['font_size_p90'] = df['font_size'] > df['font_size'].quantile(0.9)
    df['font_size_p75'] = df['font_size'] > df['font_size'].quantile(0.75)
    df['font_size_p50'] = df['font_size'] > df['font_size'].quantile(0.5)
    for col in ['font_size_p90', 'font_size_p75', 'font_size_p50']:
        X_full[col] = df[col].astype(int)
    
    # Add POS features to X_full if missing
    pos_cols = ['num_nouns', 'num_verbs', 'num_adjs', 'num_advs', 'num_propn', 'num_pronouns', 'num_other_pos']
    for col in pos_cols:
        if col not in X_full.columns:
            X_full[col] = df[col] if col in df.columns else 0
    # Use only selected features
    X = X_full[selected_features]
    # Predict heading_level (multi-class)
    df['heading_level_pred'] = clf.predict(X)
    
    # POST-PROCESSING: Filter out obvious false positives
    def is_likely_false_positive(row):
        text = str(row['text']).strip()
        
        # Filter out single words that end with punctuation (likely paragraph endings)
        if len(text.split()) == 1 and text.endswith(('.', ',', ':', ';')):
            return True
            
        # Filter out very short text (< 3 characters)
        if len(text) < 3:
            return True
            
        # Filter out text that's mostly punctuation
        if sum(1 for c in text if c in '.,;:!?-—()[]{}"\'/\\') > len(text) * 0.5:
            return True
            
        # Filter out page numbers (numbers only)
        if text.isdigit():
            return True
            
        # Filter out very long text (likely paragraphs, not headings)
        if len(text) > 200:
            return True
            
        return False
    
    # Apply filters
    df['is_false_positive'] = df.apply(is_likely_false_positive, axis=1)
    df.loc[df['is_false_positive'], 'heading_level_pred'] = 0
    
    # Only keep rows with heading_level_pred > 0
    heading_blocks = df[df['heading_level_pred'] > 0].copy()
    
    if heading_blocks.empty:
        outline = []
    else:
        # Sort by page and position for logical order
        heading_blocks = heading_blocks.sort_values(['page', 'y0'])
        
        outline = []
        for _, row in heading_blocks.iterrows():
            level = f"H{int(row['heading_level_pred'])}"
            outline.append({
                'level': level,
                'text': str(row['text']).strip(),
                'page': int(row['page'])
            })
    # Use PDF metadata title if available, else filename
    title = doc.metadata.get('title') or pdf_name
    out_path = os.path.join(OUTPUT_DIR, f'{pdf_name}.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'title': title, 'outline': outline}, f, ensure_ascii=False, indent=2)
    print(f"Predicted outline for {pdf_name} -> {out_path}")
