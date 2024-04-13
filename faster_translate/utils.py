from huggingface_hub import HfApi, snapshot_download
import os

_MODELS = {
    "banglanmt_bn2en": {
        "model_repo": "sawradip/faster-translate-banglanmt-bn2en-t5",
        "normalizer_func":"buetnlpnormalizer"
    },
    "banglanmt_en2bn": {
        "model_repo": "sawradip/faster-translate-banglanmt-en2bn-t5"
    }
}
def upload_model_hf(repo_name, folder_path, token ):
    """
    Upload all files from a folder to a Hugging Face repository.

    Parameters:
    - repo_name: The name of the Hugging Face repository (e.g., "sawradip/faster-translate-banglanmt-bn2en-t5").
    - folder_path: The local path to the folder containing the files to upload.
    """
    # Initialize a repository object, cloning it if it's not already present
    api = HfApi(token = token)

    api.upload_folder(
        folder_path=folder_path,
        repo_id=repo_name,
        repo_type="model",
    )
    
    print("All files have been uploaded.")



def download_model_hf(repo_name, save_path=None, revision=None, token=None):
    """
    Download a model from a Hugging Face repository, optionally at a specific revision, with optional save path and token.

    Parameters:
    - repo_name: The name of the Hugging Face repository (e.g., "sawradip/faster-translate-banglanmt-bn2en-t5").
    - save_path: The local path where the repository should be downloaded. If None, uses the default cache directory.
    - revision: The specific repository revision to download. If None, the latest version is downloaded.
    - token: An optional Hugging Face authentication token for private repositories.
    """
    # Use snapshot_download to download the repository or a specific snapshot of it
    if repo_name in _MODELS:
        model_args = _MODELS[repo_name]
        repo_name = model_args["model_repo"]

    downloaded_path = snapshot_download(
        repo_id=repo_name,
        cache_dir=save_path,  # If None, huggingface_hub uses its default cache directory
        revision=revision,
        use_auth_token=token if token else True  # Pass True to use the token from the HF_HOME or token cache if available
    )
    
    print(f"Model downloaded to {downloaded_path}.")
    return downloaded_path
