import os
import json
import subprocess
import ctranslate2
from tqdm.auto import tqdm
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

from .utils import download_model_hf, _MODELS


def is_cuda_available():
    # Method 1: Check using nvcc
    try:
        subprocess.check_output(['nvcc', '--version'])
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Method 2: Check CUDA_PATH environment variable
    if 'CUDA_PATH' in os.environ:
        return True

    return False

device_name = "cuda" if is_cuda_available() else "cpu"

class TranslateModel:
    def __init__(self, model_dir, source_token_list = None, normalizer_func=None, device = device_name):
        self.model_dir = model_dir
        self.translator = ctranslate2.Translator(model_dir, device=device)
        
        if normalizer_func is None:
            self.normalizer_func = lambda x: x
        elif isinstance(normalizer_func, str):
            self.normalizer_func = self.get_text_normalizer(normalizer_func)
        else:
            self.normalizer_func = normalizer_func
            
            
        if source_token_list is None:
            self.source_tokenizer_json = os.path.join(model_dir, "source_vocabulary.json")
            with open(self.source_tokenizer_json, 'r', encoding='utf-8') as file:
                source_token_list = json.load(file)
            
        self.source_tokenizer = self.get_bpe_tokenizer(source_token_list)
        
        
    def get_bpe_tokenizer(self, token_list):
        
        # Initialize a tokenizer
        tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = Whitespace()
        
        # Trainer to train the tokenizer
        trainer = BpeTrainer(special_tokens=token_list)
        tokenizer.train_from_iterator(token_list, trainer=trainer)
        
        return tokenizer
    
    
    def source_tokenize_batch(self, text_batch):
        tokenized_batch = [encoded.tokens for encoded in self.source_tokenizer.encode_batch(text_batch)]
        return tokenized_batch
    
    
    def get_text_normalizer(self, normalizer_func_name):
        if normalizer_func_name == "buetnlpnormalizer":
            try:
                from normalizer import normalize
            except:
                raise Exception("Relevant Normalizer not installed, please install with `pip install git+https://github.com/csebuetnlp/normalizer`")
            
            normalizer_func = lambda texts: [normalize(text) for text in texts]
        else:
            raise Exception(f"{normalizer_func_name} not supported yet.")
        return normalizer_func
        
        
    def detokenize_single(self, tokens):
        # Concatenate, replace special prefix, and handle edge cases
        concatenated = ''.join(tokens)
        proper_string = concatenated.replace('▁', ' ').strip()
        proper_string = proper_string.replace(' ,', ',').replace(' .', '.').replace(" '", "'").replace(" :", ":")
        return proper_string
    
    
    def detokenize_batch(self, token_batch):
        translated_decoded_batch = [self.detokenize_single(pred.hypotheses[0]) for pred in token_batch]
        return translated_decoded_batch
    
    
    def translate_single(self, text):
        text_batch = [text]
        translated_batch = self.translate_batch(text_batch)
        return translated_batch[0]


    def translate_batch(self, text_batch):
        normalized_text_batch = self.normalizer_func(text_batch)
        tokenized_text_batch = self.source_tokenize_batch(normalized_text_batch)
        translated_tokens_batch = self.translator.translate_batch(tokenized_text_batch)
        translated_batch = self.detokenize_batch(translated_tokens_batch)
        return translated_batch
    
    
    def translate_bulk(self, text_list, batch_size=1):
        text_batches = [text_list[i:i + batch_size] for i in range(0, len(text_list), batch_size)]
        
        translated_results = []
        for text_batch in tqdm(text_batches):
            translated_batch = self.translate_batch(text_batch)
            translated_results.extend(translated_batch)
        return translated_results
    
    
    def translate_file(self, file_path, batch_size, output_file = "preds.txt"):
        with open(file_path, 'r') as file:
            text_list = [line.strip() for line in file.readlines()]
            
        translated_results = self.translate_bulk(text_list, batch_size)
        
        if output_file is not None:
            translated_str = "\n".join(translated_results)
            with open(file_path, 'w') as file:
                file.write(translated_str)
                
        return translated_results
    
    
    @classmethod
    def from_pretrained(cls, model_name_or_path, save_path=None, revision=None, token=None, **kwargs):
        """
        This is for loading any pre converted translation model from huggingface.
    

        Parameters:
        - model_identifier: The name of the Hugging Face repository (e.g., "sawradip/faster-translate-banglanmt-bn2en-t5") or a local directory path.
        - save_path: The local path where the repository should be downloaded. If None, uses the default cache directory. Ignored if model_identifier is a local directory.
        - revision: The specific repository revision to download. If None, the latest version is downloaded. Ignored if model_identifier is a local directory.
        - token: An optional Hugging Face authentication token for private repositories. Ignored if model_identifier is a local directory.
        """
        
        if model_name_or_path in _MODELS:
            model_args = _MODELS[model_name_or_path]
            kwargs["normalizer_func"] = model_args.get("normalizer_func")
            
        # Check if model_identifier is a local directory
        if os.path.isdir(model_name_or_path):
            print(f"Loading model from local directory: {model_name_or_path}")
            model_path = model_name_or_path
        else:
            # Download the model using the utility function from faster_translate/utils.py
            print(f"Downloading model from Hugging Face repository: {model_name_or_path}")
            model_path = download_model_hf(model_name_or_path, save_path, revision, token)
        
        return TranslateModel(model_path, **kwargs)
        
        
            
            
            
        
            
        

            
            
        


