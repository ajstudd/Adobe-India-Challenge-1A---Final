import os
import glob
import pandas as pd
import csv

folder = 'labelled_data'
files = glob.glob(os.path.join(folder, '*.csv'))

def detect_delimiter(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        sample = f.read(2048)
        sniffer = csv.Sniffer()
        try:
            dialect = sniffer.sniff(sample)
            return dialect.delimiter
        except Exception:
            # Default to comma if detection fails
            return ','

for file in files:
    delimiter = detect_delimiter(file)
    try:
        df = pd.read_csv(file, encoding='utf-8', delimiter=delimiter)
        df.to_csv(file, index=False, encoding='utf-8-sig')
        print(f'Converted (utf-8, delimiter={delimiter!r}): {file}')
    except Exception as e:
        print(f'Failed to convert {file}: {e}')
