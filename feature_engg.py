import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import joblib
from category_encoders import TargetEncoder

# Load the dataset
df = pd.read_csv("data/bank.csv")   # apna file naam sahi kar lena
print(df.head(10))
print("Shape :", df.shape)

# Check missing values
print("Missing values:", df.isnull().sum().sum())

# Split into X and y
X = df.drop("deposit", axis=1)
y = df["deposit"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# One Hot Encoding (Safe Way)
onehot_cols = ["marital", "education", "default", "housing", "loan", "contact", "poutcome"]

X_train = pd.get_dummies(X_train, columns=onehot_cols, drop_first=True)
X_test = pd.get_dummies(X_test, columns=onehot_cols, drop_first=True)

# Target Encoding for high cardinality columns

job_encoder = TargetEncoder()
month_encoder = TargetEncoder()

X_train['job'] = job_encoder.fit_transform(X_train['job'], y_train)
X_test['job'] = job_encoder.transform(X_test['job'])

X_train['month'] = month_encoder.fit_transform(X_train['month'], y_train)
X_test['month'] = month_encoder.transform(X_test['month'])

# Align columns to avoid mismatch
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# Convert to float
X_train = X_train.astype(float)
X_test = X_test.astype(float)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# Save the preprocessed data
joblib.dump(X_train, 'models/X_train.pkl')
joblib.dump(X_test, 'models/X_test.pkl')
joblib.dump(y_train, 'models/y_train.pkl')
joblib.dump(y_test, 'models/y_test.pkl')


# Save Encoders
joblib.dump(job_encoder, 'models/job_encoder.pkl')
joblib.dump(month_encoder, 'models/month_encoder.pkl')

# Save training columns for alignment

joblib.dump(X_train.columns.tolist(), 'models/train_columns.pkl')
print("Train columns saved for alignment!")

print("All files saved successfully!")