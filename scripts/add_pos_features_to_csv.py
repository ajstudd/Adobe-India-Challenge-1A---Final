import pandas as pd
import sys
from langdetect import detect
import spacy
import os

"""
This script adds POS-based features to an existing labelled CSV.
It only processes English rows (detected by langdetect).
POS features added: num_nouns, num_verbs, num_adjs, num_advs, num_propn, num_pronouns, num_other_pos
Usage:
    python add_pos_features_to_csv.py input.csv output.csv
"""

def ensure_spacy_model():
    try:
        nlp = spacy.load('en_core_web_sm')
    except OSError:
        print('Downloading spaCy English model...')
        from spacy.cli import download
        download('en_core_web_sm')
        nlp = spacy.load('en_core_web_sm')
    return nlp

def add_pos_features(df, text_col='text'):
    nlp = ensure_spacy_model()
    pos_counts = {
        'num_nouns': [],
        'num_verbs': [],
        'num_adjs': [],
        'num_advs': [],
        'num_propn': [],
        'num_pronouns': [],
        'num_other_pos': []
    }
    for text in df[text_col].astype(str):
        try:
            lang = detect(text)
        except Exception:
            lang = 'unknown'
        if lang == 'en':
            doc = nlp(text)
            noun = sum(1 for t in doc if t.pos_ == 'NOUN')
            verb = sum(1 for t in doc if t.pos_ == 'VERB')
            adj = sum(1 for t in doc if t.pos_ == 'ADJ')
            adv = sum(1 for t in doc if t.pos_ == 'ADV')
            propn = sum(1 for t in doc if t.pos_ == 'PROPN')
            pron = sum(1 for t in doc if t.pos_ == 'PRON')
            other = len(doc) - (noun + verb + adj + adv + propn + pron)
        else:
            noun = verb = adj = adv = propn = pron = other = 0
        pos_counts['num_nouns'].append(noun)
        pos_counts['num_verbs'].append(verb)
        pos_counts['num_adjs'].append(adj)
        pos_counts['num_advs'].append(adv)
        pos_counts['num_propn'].append(propn)
        pos_counts['num_pronouns'].append(pron)
        pos_counts['num_other_pos'].append(other)
    for k, v in pos_counts.items():
        df[k] = v
    return df


def main():
    labelled_dir = 'labelled_data'
    csv_files = [f for f in os.listdir(labelled_dir) if f.endswith('.csv')]
    if not csv_files:
        print('No CSV files found in labelled_data.')
        return
    for fname in csv_files:
        path = os.path.join(labelled_dir, fname)
        print(f'Processing {path}...')
        df = pd.read_csv(path)
        df2 = add_pos_features(df)
        df2.to_csv(path, index=False, encoding='utf-8-sig')
        print(f'POS features added and saved to {path}')

if __name__ == '__main__':
    main()
