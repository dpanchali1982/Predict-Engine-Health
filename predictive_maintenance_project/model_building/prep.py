
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi, login

# 1. Authentication & Configuration
hf_token = os.getenv("HF_TOKEN")
api = HfApi(token=hf_token)

REPO_ID = "dpanchali/predictive_maintenance"
DATASET_PATH = f"hf://datasets/{REPO_ID}/engine_data.csv"

# 2. Data Loading & Initial Cleaning
print("Loading dataset...")
df = pd.read_csv(DATASET_PATH)
initial_count = len(df)

# Basic Cleaning
df = df.dropna().drop_duplicates()

# Outlier Removal using IQR Method
def clean_outliers(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

cols_to_filter = ['Coolant temp', 'Fuel pressure']
df_cleaned = clean_outliers(df, cols_to_filter)

print(f"Cleaning complete. Records: {initial_count} -> {len(df_cleaned)}")

# 3. Data Splitting
X = df_cleaned.drop(columns=['Engine Condition'])
y = df_cleaned['Engine Condition']

# 80/20 split with stratification to preserve the target class ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Save and Upload
files = {
    "X_train.csv": X_train,
    "X_test.csv": X_test,
    "y_train.csv": y_train,
    "y_test.csv": y_test
}

print("Saving and uploading files...")
for filename, data in files.items():
    data.to_csv(filename, index=False)
    try:
        api.upload_file(
            path_or_fileobj=filename,
            path_in_repo=filename,
            repo_id=REPO_ID,
            repo_type="dataset"
        )
        print(f"✅ Uploaded: {filename}")
    except Exception as e:
        print(f"❌ Error uploading {filename}: {e}")

print(f"\nPipeline finished. Dataset view: https://huggingface.co/datasets/{REPO_ID}")
