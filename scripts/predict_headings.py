import pandas as pd
import joblib

# Load trained model
clf = joblib.load('../models/heading_classifier.pkl')

# Load extracted blocks from your PDFs
blocks_path = '../output/your_extracted_blocks.csv'
df = pd.read_csv(blocks_path)
features = ['x0', 'y0', 'x1', 'y1', 'page']
X = df[features]
df['is_heading_pred'] = clf.predict(X)
df.to_csv('../output/labeled_blocks.csv', index=False)
print('Predictions saved to ../output/labeled_blocks.csv')
