import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFECV
from sklearn.metrics import roc_auc_score, roc_curve, auc
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
import glob
import os
from sklearn.preprocessing import LabelEncoder

# Load all labelled CSVs from labelled_data folder
csv_dir = 'labelled_data'
csv_files = glob.glob(os.path.join(csv_dir, '*.csv'))

dfs = [pd.read_csv(f) for f in csv_files]
df = pd.concat(dfs, ignore_index=True)


# Drop rows with NaN in any feature columns (except those that can be None)
feature_cols = [
    'x0', 'y0', 'x1', 'y1', 'page', 'font_size', 'font', 'color',
    'bold', 'italic', 'underline', 'bullet', 'math', 'hyperlink',
    'is_all_caps', 'is_title_case', 'ends_with_colon', 'starts_with_number',
    'punctuation_count', 'line_position_on_page', 'relative_font_size',
    'distance_to_previous_heading', 'line_spacing_above', 'text', 'is_heading'
]
df = df.dropna(subset=['x0', 'y0', 'x1', 'y1', 'page', 'font_size', 'text', 'is_heading'])
df['text'] = df['text'].astype(str)

# Encode categorical features
for col in ['font', 'color']:
    if col in df.columns:
        df[col] = df[col].astype(str).fillna('None')
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

# Convert boolean features to int
for col in ['bold', 'italic', 'underline', 'bullet', 'math', 'hyperlink',
            'is_all_caps', 'is_title_case', 'ends_with_colon', 'starts_with_number']:
    if col in df.columns:
        df[col] = df[col].astype(float).fillna(0).astype(int)



# Fill missing numeric features with 0
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



# Add text-based features
df['text_length'] = df['text'].apply(len)

# Define common heading keywords
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

# Feature list (all engineered features)
features = [
    'x0', 'y0', 'x1', 'y1', 'page', 'font_size', 'font', 'color',
    'bold', 'italic', 'underline', 'bullet', 'math', 'hyperlink',
    'is_all_caps', 'is_title_case', 'ends_with_colon', 'starts_with_number',
    'punctuation_count', 'contains_colon', 'contains_semicolon', 'word_count',
    'line_position_on_page', 'relative_font_size', 'distance_to_previous_heading', 'line_spacing_above',
    'text_length', 'contains_keyword'
]
# Add POS features if present
for col in pos_cols:
    if col in df.columns and col not in features:
        features.append(col)


# Utility: Combine consecutive headings with same heading_level

def combine_consecutive_headings(df):
    # Only combine rows where is_heading==1 and heading_level > 0
    combined = []
    prev_row = None
    for idx, row in df.iterrows():
        if row['is_heading'] == 1 and row['heading_level'] > 0:
            if prev_row is not None and prev_row['is_heading'] == 1 and prev_row['heading_level'] == row['heading_level']:
                # Merge text
                prev_row['text'] += ' ' + row['text']
            else:
                if prev_row is not None:
                    combined.append(prev_row)
                prev_row = row.copy()
        else:
            if prev_row is not None:
                combined.append(prev_row)
                prev_row = None
            combined.append(row)
    if prev_row is not None:
        combined.append(prev_row)
    return pd.DataFrame(combined)


# Always set heading_level to 0 if is_heading is 0
df.loc[df['is_heading'] == 0, 'heading_level'] = 0
print('Training multi-class classifier (heading_level: 0=not heading, 1=H1, 2=H2, 3=H3, 4=H4)')
# Combine consecutive headings with same heading_level
df = combine_consecutive_headings(df)
y = df['heading_level'].fillna(0).astype(int)



# TF-IDF features
tfidf_vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['text'])
tfidf_feature_names = [f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])]
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_feature_names, index=df.index)

# Concatenate TF-IDF features to X
X = pd.concat([df[features], tfidf_df], axis=1)



# Dynamic font thresholding: use relative font size (already present), but also add per-document font size percentiles
df['font_size_p90'] = df['font_size'] > df['font_size'].quantile(0.9)
df['font_size_p75'] = df['font_size'] > df['font_size'].quantile(0.75)
df['font_size_p50'] = df['font_size'] > df['font_size'].quantile(0.5)
for col in ['font_size_p90', 'font_size_p75', 'font_size_p50']:
    X[col] = df[col].astype(int)
    if col not in features:
        features.append(col)

# SMOTE class balancing
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Feature selection with RFECV
clf_base = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rfecv = RFECV(estimator=clf_base, step=1, cv=3, scoring='accuracy', n_jobs=-1)
rfecv.fit(X_res, y_res)
print(f'Optimal number of features: {rfecv.n_features_}')
selected_features = list(X.columns[rfecv.support_])
X_selected = X_res[selected_features]

# Train/test split on balanced, selected features
X_train, X_test, y_train, y_test = train_test_split(X_selected, y_res, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
clf.fit(X_train, y_train)
print('Test accuracy:', clf.score(X_test, y_test))

# Save model, vectorizer, and selected features
joblib.dump(clf, 'models/docbank_heading_classifier_local.pkl')
joblib.dump(tfidf_vectorizer, 'models/docbank_heading_classifier_local_tfidf_vectorizer.pkl')
joblib.dump(selected_features, 'models/docbank_heading_classifier_local_selected_features.pkl')
print('Model saved to models/docbank_heading_classifier_local.pkl')
print('TF-IDF vectorizer saved to models/docbank_heading_classifier_local_tfidf_vectorizer.pkl')
print('Selected features saved to models/docbank_heading_classifier_local_selected_features.pkl')

# Show class distribution
print('Class distribution in full data:')
print(y.value_counts())


# Evaluate on test set
y_pred = clf.predict(X_test)
print('Confusion matrix:')
print(confusion_matrix(y_test, y_pred))
print('Classification report:')
print(classification_report(y_test, y_pred))

# ROC & AUC evaluation (one-vs-rest for multi-class)
try:
    from sklearn.preprocessing import label_binarize
    y_test_bin = label_binarize(y_test, classes=sorted(set(y)))
    y_score = clf.predict_proba(X_test)
    if y_test_bin.shape[1] == 1:
        auc_score = roc_auc_score(y_test, y_score[:, 1])
        print(f'ROC AUC (binary): {auc_score:.4f}')
    else:
        auc_score = roc_auc_score(y_test_bin, y_score, average='macro', multi_class='ovr')
        print(f'ROC AUC (multi-class, macro): {auc_score:.4f}')
except Exception as e:
    print(f'Could not compute ROC AUC: {e}')

# Show sample predictions
sample = X_test.head(10)
preds = clf.predict(sample)
print('Sample predictions:', preds)
