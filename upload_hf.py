from huggingface_hub import HfApi

REPO = "muskan336/indic-manuscripts-synthetic"

api = HfApi()
api.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)
api.upload_folder(
    folder_path="hf_dataset",
    repo_id=REPO,
    repo_type="dataset",
    commit_message="Initial upload: Devanagari, Modi, Sharada synthetic manuscripts",
)
print("done: https://huggingface.co/datasets/" + REPO)