import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load mapped DocBank CSV
csv_path = '../dataset/mapped_docbank.csv'
df = pd.read_csv(csv_path)
features = ['x0', 'y0', 'x1', 'y1', 'page']
X = df[features]
y = df['is_heading']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
print('Test accuracy:', clf.score(X_test, y_test))

joblib.dump(clf, '../models/heading_classifier.pkl')
print('Model saved to ../models/heading_classifier.pkl')
