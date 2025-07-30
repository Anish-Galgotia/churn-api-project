# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# Load dataset
df = pd.read_csv('telco_churn.csv')

# Drop ID column
df.drop('customerID', axis=1, inplace=True)

# Convert TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Handle missing values (if any)
df.dropna(inplace=True)

# Print status
# print("Data shape:", df.shape)
# print("Missing values:", df.isnull().sum())

# Target variable
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Separate features
X = df.drop('Churn', axis=1)
y = df['Churn']

# Identify categorical and numerical columns
cat_cols = X.select_dtypes(include='object').columns.tolist()
num_cols = X.select_dtypes(include='number').columns.tolist()

# One-hot encode categorical columns
X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True, dtype=int)

# Scale numeric columns
scaler = StandardScaler()
X_encoded[num_cols] = scaler.fit_transform(X_encoded[num_cols])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model and assets
joblib.dump(model, 'model/churn_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')
joblib.dump(X_encoded.columns.tolist(), 'model/feature_names.pkl')
joblib.dump(num_cols, 'model/numeric_cols.pkl')

print("Model and preprocessing files saved.")



