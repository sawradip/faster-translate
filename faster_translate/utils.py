from huggingface_hub import HfApi, snapshot_download
import os

# Dictionary of predefined models with their configurations
_MODELS = {
    "banglanmt_bn2en": {
        "model_repo": "sawradip/faster-translate-banglanmt-bn2en-t5",
        "normalizer_func": "buetnlpnormalizer",
        "model_type": "ctranslate2"
    },
    "banglanmt_en2bn": {
        "model_repo": "sawradip/faster-translate-banglanmt-en2bn-t5",
        "model_type": "ctranslate2"
    },
    "bangla_mbartv1_en2bn": {
        "model_repo": "sawradip/faster-translate-banglabart-en2bn-v1",
        "tokenizer_repo": "facebook/mbart-large-50-many-to-many-mmt",
        "model_type": "ctranslate2"
    },
    "bangla_qwen_en2bn": {
        "model_repo": "AI4BD/Bangla-Qwen-Translator-v1.2",
        "tokenizer_repo": "AI4BD/Bangla-Qwen-Translator-Tokenizer",
        "model_type": "vllm",
        "max_model_len": 4096,
        "gpu_memory_utilization": 0.95,
        "dtype": "bfloat16"
    }
}

def download_model_hf(repo_id, local_dir=None, revision=None, token=None):
    """
    Download a model from Hugging Face Hub.
    
    Args:
        repo_id: Repository ID on Hugging Face Hub
        local_dir: Local directory to save the model
        revision: Specific revision to download
        token: HuggingFace token for private repositories
        
    Returns:
        Path to the downloaded model
    """
    if repo_id in _MODELS:
        repo_id = _MODELS[repo_id]["model_repo"]
    
    print(f"Downloading model from Hugging Face Hub: {repo_id}")
    
    kwargs = {
        "repo_id": repo_id,
        "local_dir": local_dir,
    }
    
    if revision:
        kwargs["revision"] = revision
    if token:
        kwargs["token"] = token
    
    model_path = snapshot_download(**kwargs)
    print(f"Model downloaded to: {model_path}")
    
    return model_path

def add_model(model_name, model_config):
    """
    Add a new model to the _MODELS dictionary.
    
    Args:
        model_name: Short name for the model
        model_config: Dictionary with model configuration
        
    Returns:
        None
    """
    if model_name in _MODELS:
        print(f"Warning: Overwriting existing model configuration for {model_name}")
    
    _MODELS[model_name] = model_config
    print(f"Added model {model_name} to available models")