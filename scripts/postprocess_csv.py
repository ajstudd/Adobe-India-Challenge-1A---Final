import pandas as pd
import sys

"""
This script postprocesses a CSV after manual heading labeling.
For each row with is_heading==1, if the text contains a punctuation (.,:;), split at the first punctuation:
 - Left part remains as heading (is_heading=1, heading_level as before)
 - Right part becomes a new row (is_heading=0, heading_level=0), inserted after the heading row
Also fills the 'distance_to_previous_heading' column.
Usage:
    python postprocess_csv.py input.csv output.csv
"""

def split_heading_row(row):
    import re
    text = str(row['text'])
    # Regex: match punctuation (.,:;) not preceded and followed by a digit (to avoid 1.1, 2.3.4, etc)
    # Also allow for section labels like (1.1)
    # This will match the first punctuation that is NOT between digits
    match = re.search(r'(?<!\d)[.,:;](?!\d)', text)
    if match:
        idx = match.start()
        left_text = text[:idx+1].strip()  # include the punctuation in the left part
        right_text = text[idx+1:].strip()
        left_row = row.copy()
        left_row['text'] = left_text
        right_row = row.copy()
        right_row['text'] = right_text
        right_row['is_heading'] = 0
        right_row['heading_level'] = 0
        return [left_row, right_row] if right_text else [left_row]
    return [row]

def postprocess_csv(df):
    # Split heading rows as needed
    new_rows = []
    for _, row in df.iterrows():
        split_rows = split_heading_row(row)
        new_rows.extend(split_rows)
    df2 = pd.DataFrame(new_rows)
    # Recompute distance_to_previous_heading
    last_heading_idx = None
    distances = []
    for idx, is_heading in enumerate(df2['is_heading']):
        if is_heading == 1 or is_heading == '1':
            distances.append(0)
            last_heading_idx = idx
        elif last_heading_idx is not None:
            distances.append(idx - last_heading_idx)
        else:
            distances.append(None)
    df2['distance_to_previous_heading'] = distances
    return df2

import os
import glob

def process_file(input_csv):
    df = pd.read_csv(input_csv)
    df2 = postprocess_csv(df)
    # Overwrite the original file in place
    df2.to_csv(input_csv, index=False, encoding='utf-8-sig')
    print(f"Postprocessed and saved to {input_csv}")

if __name__ == "__main__":
    DEFAULT_INDIR = 'labelled_data'
    if len(sys.argv) == 1:
        # No arguments: process all CSVs in labelled_data and overwrite them in place
        input_folder = DEFAULT_INDIR
        csv_files = glob.glob(os.path.join(input_folder, '*.csv'))
        for input_csv in csv_files:
            process_file(input_csv)
        print(f"Batch processing complete. Processed {len(csv_files)} files.")
    elif len(sys.argv) == 2:
        # Single file mode: overwrite in place
        input_csv = sys.argv[1]
        process_file(input_csv)
    elif len(sys.argv) == 3 and sys.argv[1] == '--batch':
        # Batch mode: python postprocess_csv.py --batch input_folder
        input_folder = sys.argv[2]
        csv_files = glob.glob(os.path.join(input_folder, '*.csv'))
        for input_csv in csv_files:
            process_file(input_csv)
        print(f"Batch processing complete. Processed {len(csv_files)} files.")
    else:
        print("Usage:")
        print("  No args:    python postprocess_csv.py (process all in labelled_data, overwrite in place)")
        print("  Single file: python postprocess_csv.py input.csv (overwrite in place)")
        print("  Batch mode:  python postprocess_csv.py --batch input_folder (overwrite in place)")
        sys.exit(1)
