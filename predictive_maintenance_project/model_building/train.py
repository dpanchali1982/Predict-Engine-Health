import os
import joblib
import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# ==========================================
# 1. Configuration & Setup
# ==========================================
# GitHub runners don't have a server, so we use a local file-based URI
mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))
mlflow.set_experiment("predictive-maintenance-experiment")

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)
api = HfApi(token=HF_TOKEN)

# ==========================================
# 2. Load Processed Data from Hugging Face
# ==========================================
REPO_ID = "dpanchali/predictive_maintenance"
BASE_URL = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main"

print("Loading data from Hugging Face...")
X_train = pd.read_csv(f"{BASE_URL}/X_train.csv")
X_test = pd.read_csv(f"{BASE_URL}/X_test.csv")
y_train = pd.read_csv(f"{BASE_URL}/y_train.csv").values.ravel()
y_test = pd.read_csv(f"{BASE_URL}/y_test.csv").values.ravel()

# ==========================================
# 3. Custom Feature Engineering (Pipeline Style)
# ==========================================
def engineer_features(df):
    X = df.copy()
    X['load_index'] = X['Engine rpm'] * X['Fuel pressure']
    X['thermal_stress'] = X['Coolant temp'] / (X['Lub oil pressure'] + 1e-5)
    return X

# Wrap the function for the Scikit-Learn pipeline
feature_transformer = FunctionTransformer(engineer_features)

# ==========================================
# 4. Model Definition & Hyperparameter Tuning
# ==========================================
print("Initializing XGBoost and Pipeline...")

xgb_clf = xgb.XGBClassifier(
    scale_pos_weight=1.7, # Based on your ~37% faulty cases
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

# Create the full pipeline: Feature Engineering -> Model
# This is exactly like your Tourism project's 'make_pipeline'
model_pipeline = make_pipeline(feature_transformer, xgb_clf)

param_grid = {
    'xgbclassifier__n_estimators': [100, 200, 300],
    'xgbclassifier__max_depth': [3, 5, 7],
    'xgbclassifier__learning_rate': [0.01, 0.1, 0.2]
}

# Start MLflow Run
with mlflow.start_run(run_name="XGBoost_Maintenance_Final"):
    print("Training model pipeline with Random Search...")
    
    random_search = RandomizedSearchCV(
        model_pipeline, 
        param_distributions=param_grid, 
        n_iter=5, 
        cv=3, 
        scoring='f1'
    )
    
    random_search.fit(X_train, y_train)
    best_pipeline = random_search.best_estimator_

    # ==========================================
    # 5. Evaluation
    # ==========================================
    y_pred = best_pipeline.predict(X_test)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred)
    }

    print("\nFINAL PERFORMANCE METRICS:")
    for k, v in metrics.items():
        print(f"  {k.capitalize()}: {v:.4f}")

    # ==========================================
    # 6. Logging & Saving
    # ==========================================
    mlflow.log_params(random_search.best_params_)
    mlflow.log_metrics(metrics)

    # Save the ENTIRE pipeline (including feature engineering)
    model_filename = "predictive_maintenance_model.joblib"
    joblib.dump(best_pipeline, model_filename)
    
    # Log artifact to MLflow
    mlflow.log_artifact(model_filename, artifact_path="model")

    # ==========================================
    # 7. Upload to Hugging Face
    # ==========================================
    MODEL_REPO_ID = "dpanchali/predictive_maintenance_model"
    if HF_TOKEN:
        try:
            try:
                api.repo_info(repo_id=MODEL_REPO_ID, repo_type="model")
            except RepositoryNotFoundError:
                create_repo(repo_id=MODEL_REPO_ID, repo_type="model", private=False)

            api.upload_file(
                path_or_fileobj=model_filename,
                path_in_repo=model_filename,
                repo_id=MODEL_REPO_ID,
                repo_type="model"
            )
            print("✅ Success! Model uploaded to Hugging Face.")
        except Exception as e:
            print(f"Error during HF upload: {e}")

print("Training pipeline completed.")
