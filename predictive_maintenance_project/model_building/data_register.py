import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# 1. Configuration
repo_id = "dpanchali/predictive_maintenance"
repo_type = "dataset"
folder_path = "predictive_maintenance_project/data"

# 2. Initialize API client using environment variable
hf_token = os.getenv("HF_TOKEN")
api = HfApi(token=hf_token)

# 3. Check if the repository exists, create if not
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Repo '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Repo '{repo_id}' not found. Creating new repo...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Repo '{repo_id}' created.")

# 4. Upload the folder
print(f"Uploading data from {folder_path}...")
try:
    api.upload_folder(
        folder_path=folder_path,
        repo_id=repo_id,
        repo_type=repo_type,
    )
    print(f"✅ Success! Raw dataset registered at: https://huggingface.co/datasets/{repo_id}")
except FileNotFoundError:
    print(f"❌ Error: The folder '{folder_path}' was not found. Please check your path.")
except Exception as e:
    print(f"❌ An unexpected error occurred: {e}")
