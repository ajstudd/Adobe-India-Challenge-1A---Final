import pandas as pd
import sys

"""
This script fills the 'distance_to_previous_heading' column in a CSV after manual heading labeling.
Usage:
    python populate_distance_to_previous_heading.py input.csv output.csv
"""

def populate_distance_to_previous_heading(df, heading_col='is_heading', distance_col='distance_to_previous_heading'):
    last_heading_idx = None
    distances = []
    for idx, is_heading in enumerate(df[heading_col]):
        if is_heading == 1 or is_heading == '1':
            distances.append(0)
            last_heading_idx = idx
        elif last_heading_idx is not None:
            distances.append(idx - last_heading_idx)
        else:
            distances.append(None)
    df[distance_col] = distances
    return df

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python populate_distance_to_previous_heading.py input.csv output.csv")
        sys.exit(1)
    input_csv, output_csv = sys.argv[1], sys.argv[2]
    df = pd.read_csv(input_csv)
    df = populate_distance_to_previous_heading(df)
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"Populated distance_to_previous_heading and saved to {output_csv}")
