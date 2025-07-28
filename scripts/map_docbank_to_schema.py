import os
import csv
import unicodedata
from src.utils.text_utils import contains_mojibake, clean_text

def map_docbank_to_schema(docbank_txt_folder, output_csv):
    # Only keep 'title' and 'section' blocks, map heading level
    with open(output_csv, 'w', newline='', encoding='utf-8') as out_f:
        fieldnames = [
            'text', 'is_heading', 'heading_level', 'page', 'x0', 'y0', 'x1', 'y1'
        ]
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
import os
import csv

def map_docbank_to_schema(docbank_txt_folder, output_csv):
    # Only keep 'title' and 'section' blocks, map heading level
    with open(output_csv, 'w', newline='', encoding='utf-8') as out_f:
        fieldnames = [
            'text', 'is_heading', 'heading_level', 'page', 'x0', 'y0', 'x1', 'y1'
        ]
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        # Resume support: check for processed files log
        log_path = output_csv + '.log'
        processed_files = set()
        if os.path.exists(log_path):
            with open(log_path, 'r') as logf:
                processed_files = set(line.strip() for line in logf)
        # If output_csv exists, do not write header again
        write_header = not os.path.exists(output_csv) or os.path.getsize(output_csv) == 0
        if write_header:
            writer.writeheader()
        for fname in sorted(os.listdir(docbank_txt_folder)):
            if not fname.endswith('.txt'):
                continue
            if fname in processed_files:
                continue
            # Try to extract page number from filename, fallback to None
            try:
                page_num = int(fname.split('_')[-1].replace('.txt', ''))
            except Exception:
                page_num = None
            # Read all blocks and merge by label, page, y0
            blocks = []
            with open(os.path.join(docbank_txt_folder, fname), 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) < 10:
                        continue
                    text = parts[0]
                    if text.startswith('##LTLine##'):
                        continue  # Skip LTLine entries
                    x0, y0, x1, y1 = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    label = parts[-1]
            # Clean and filter text
            text = clean_text(text)
            text = unicodedata.normalize("NFKC", text)
            if not text or contains_mojibake(text):
                continue
            blocks.append({
                'text': text,
                'label': label,
                'page': page_num,
                'x0': x0,
                'y0': y0,
                'x1': x1,
                'y1': y1
            })
            # Merge consecutive blocks with same label, page, y0
            merged = []
            if not blocks:
                # Mark file as processed even if empty
                with open(log_path, 'a') as logf:
                    logf.write(fname + '\n')
                continue
            prev = None
            for block in blocks:
                key = (block['label'], block['page'], block['y0'])
                if prev and (prev['label'], prev['page'], prev['y0']) == key:
                    # Merge text and expand x1
                    prev['text'] += ' ' + block['text']
                    prev['x1'] = max(prev['x1'], block['x1'])
                    prev['y1'] = max(prev['y1'], block['y1'])
                else:
                    if prev:
                        merged.append(prev)
                    prev = block.copy()
            if prev:
                merged.append(prev)
            # Write merged blocks
            for block in merged:
                text = block['text'].strip()
                # Heuristic for headings
                is_heading = 0
                heading_level = ''
                # Detect numbered headings (e.g., 1., 1.1, 2.3.4)
                import re
                numbered = re.match(r'^(\d+(\.\d+)*)(\s+|\.|:)', text)
                if block['label'] == 'title' or (numbered and len(text) < 100):
                    is_heading = 1
                    heading_level = 'H1' if block['label'] == 'title' else 'H2'
                elif block['label'] == 'section' and len(text) < 100:
                    is_heading = 1
                    heading_level = 'H2'
                # Font/formatting features (if available)
                # TODO: Add more features if present in DocBank
                writer.writerow({
                    'text': text,
                    'is_heading': is_heading,
                    'heading_level': heading_level,
                    'page': block['page'],
                    'x0': block['x0'],
                    'y0': block['y0'],
                    'x1': block['x1'],
                    'y1': block['y1']
                })
            # Mark file as processed
            with open(log_path, 'a') as logf:
                logf.write(fname + '\n')

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print('Usage: python scripts/map_docbank_to_schema.py <docbank_txt_folder>')
        print('Output will be saved to output/mapped_docbank.csv')
    else:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_csv = os.path.join(repo_root, 'output', 'mapped_docbank.csv')
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        map_docbank_to_schema(sys.argv[1], output_csv)
