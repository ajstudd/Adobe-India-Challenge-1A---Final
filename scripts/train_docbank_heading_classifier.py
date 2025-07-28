import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import classification_report, confusion_matrix

# Load mapped DocBank CSV
csv_path = 'output/mapped_docbank.csv'
df = pd.read_csv(csv_path)
# Drop rows with NaN in any feature columns
df = df.dropna(subset=['x0', 'y0', 'x1', 'y1', 'page', 'text', 'is_heading'])
df['text'] = df['text'].astype(str)
# Basic feature engineering
df['text_len'] = df['text'].apply(len)
df['num_words'] = df['text'].apply(lambda x: len(str(x).split()))
features = ['x0', 'y0', 'x1', 'y1', 'page', 'text_len', 'num_words']
X = df[features]
y = df['is_heading']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
clf.fit(X_train, y_train)
print('Test accuracy:', clf.score(X_test, y_test))

joblib.dump(clf, 'models/docbank_heading_classifier.pkl')
print('Model saved to models/docbank_heading_classifier.pkl')

# Show class distribution
print('Class distribution in full data:')
print(df['is_heading'].value_counts())

# Evaluate on test set
y_pred = clf.predict(X_test)
print('Confusion matrix:')
print(confusion_matrix(y_test, y_pred))
print('Classification report:')
print(classification_report(y_test, y_pred))

# Show sample predictions
sample = X_test.head(10)
preds = clf.predict(sample)
print('Sample predictions:', preds)
