import os
import json
import subprocess
import ctranslate2
from tqdm.auto import tqdm
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from transformers import AutoTokenizer
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
    def __init__(self, model_dir, source_token_list = None, tokenizer_filename = None, tokenizer_repo = None, normalizer_func=None, device = device_name):
        self.model_dir = model_dir
        self.translator = ctranslate2.Translator(model_dir, device=device)
        
        if normalizer_func is None:
            self.normalizer_func = lambda x: x
        elif isinstance(normalizer_func, str):
            self.normalizer_func = self.get_text_normalizer(normalizer_func)
        else:
            self.normalizer_func = normalizer_func
            
        self.source_tokenizer = self.get_hf_tokenizer(tokenizer_repo)
        if self.source_tokenizer is None:
            if source_token_list is None:
                source_token_list = self.get_token_list(tokenizer_filename)
            self.source_tokenizer = self.get_bpe_tokenizer(source_token_list)
        
    def get_hf_tokenizer(self, tokenizer_repo):
        if tokenizer_repo is not None:
            return  AutoTokenizer.from_pretrained(tokenizer_repo)
        elif os.path.isfile(os.path.join(self.model_dir, 'tokenizer_config.json')):
            return  AutoTokenizer.from_pretrained(self.model_dir)
        else:
            return None       
         
    def get_token_list(self, tokenizer_filename):
        
        if tokenizer_filename is not None:
            tokenizer_filepath  = os.path.join(self.model_dir, tokenizer_filename)
        elif os.path.isfile(os.path.join(self.model_dir, "source_vocabulary.json")):
            tokenizer_filepath  = os.path.join(self.model_dir, "source_vocabulary.json")
        elif os.path.isfile(os.path.join(self.model_dir, "shared_vocabulary.json")):
            tokenizer_filepath  = os.path.join(self.model_dir, "shared_vocabulary.json")
        else:
            raise Exception("Vocabulary file was not found.")
        
        with open(tokenizer_filepath, 'r', encoding='utf-8') as file:
            source_token_list = json.load(file)
        return source_token_list
    

        
    def get_bpe_tokenizer(self, token_list):
        
        # Initialize a tokenizer
        tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = Whitespace()
        
        # Trainer to train the tokenizer
        trainer = BpeTrainer(special_tokens=token_list, vocab_size=len(token_list), min_frequency=1)
        tokenizer.train_from_iterator(token_list, trainer=trainer)
        
        return tokenizer
    
    
    def source_tokenize_batch(self, text_batch):
        if isinstance(self.source_tokenizer, Tokenizer):
            text_batch = [f" {text}".replace(" ", "▁") for text in text_batch]
            tokenized_batch = [encoded.tokens for encoded in self.source_tokenizer.encode_batch(text_batch)]
        else:
            tokenized_batch = [self.source_tokenizer.convert_ids_to_tokens(
                                    input_id_list) for input_id_list in self.source_tokenizer(text_batch)["input_ids"]]
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
        if isinstance(self.source_tokenizer, Tokenizer):
            concatenated = ''.join(tokens)
            proper_string = concatenated.replace('▁', ' ').strip()
            proper_string = proper_string.replace(' ,', ',').replace(' .', '.').replace(" '", "'").replace(" :", ":")
        else:
            proper_string = self.source_tokenizer.decode(self.source_tokenizer.convert_tokens_to_ids(tokens), skip_special_tokens=True)
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

    def translate_hf_dataset(self, 
                             dataset_repo, 
                             subset_name = None,
                             split=["train"], 
                             columns=[], 
                             batch_size=1, 
                             token = None, 
                             start_idx = 0,
                             end_idx = -1,
                             output_format = "json",
                             output_name="preds.json",):

        dataset_args = [dataset_repo]
        if subset_name is not None:
            dataset_args.append(subset_name)
            
        dataset_kwargs = {}
        if token is not None:
            dataset_kwargs["token"] = token
        dataset = load_dataset(*dataset_args, **dataset_kwargs)

        
        if split == "*":
            split = [split for split in dataset.keys()]
        if isinstance(split, str):
            split = [split]
        
        final_dataset_dict = {}
        for split_name in split:
            split_data = dataset[split_name]
            
            flattened_data_list = []
            data_length_map = []
            final_dataset_dict[split_name] = {}
            for column in columns:
                data_list = split_data[column]
                
                if isinstance(data_list[0], list) or (data_list[0].startswith("[") and data_list[0].endsswith("]") and isinstance(eval(data_list[0]), list)):
                    for sample in data_list[start_idx:end_idx]:
                        sample = sample if isinstance(data_list[0], list) else eval(sample)
                        data_length_map.append(len(sample))
                        flattened_data_list.extend(sample)
                elif isinstance(data_list[0], str):
                    flattened_data_list = data_list[start_idx:end_idx]
                else:
                    raise Exception(f"We only support of `str` or `List[str]` type columns,  not `{type(data_list[0])}`")
                
                translated_flattened_data_list = self.translate_bulk(flattened_data_list, batch_size=batch_size)
                
                index_ptr = 0
                translated_data_list = []
                if data_length_map == []:
                    translated_data_list = translated_flattened_data_list
                for data_length in data_length_map:
                    translated_data_list.append(translated_flattened_data_list[index_ptr: index_ptr+data_length])
                    index_ptr += data_length
                    
                final_dataset_dict[split_name][column] = translated_data_list
                
        with open(output_name, 'w', encoding='utf-8') as f:
            json.dump(final_dataset_dict, f, ensure_ascii=False, indent=4)
            
        return final_dataset_dict
            
            
                
            
                    
                
                    
                        
                    
                    
                
            
        
        # text_list = dataset[split][column]
    
        # # Subset the text list if start_index and end_index are provided
        # if start_index is not None and end_index is not None:
        #     text_list = text_list[start_index:end_index+1]  # Add 1 to end_index to include it in the range
    
        # # Translate in batches
        # translated_results=[]
        # for text in text_list:
        #     translated_batch=self.translate_batch1(text,batch_size=batch_size)
        #     translated_results.append(translated_batch)
        # return translated_results
            
        # translated_results = []
        # for i in range(0, len(text_list), batch_size):
        #     batch = text_list[i:i + batch_size]
        #     translated_batch = self.translate_batch(batch)
        #     translated_results.append(translated_batch)
        # return translated_results
    @classmethod
    def from_pretrained(cls, model_name_or_path, save_path=None, revision=None, token=None, tokenizer_repo = None, **kwargs):
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
            kwargs["tokenizer_repo"] = tokenizer_repo if tokenizer_repo else model_args.get("tokenizer_repo")
            
        # Check if model_identifier is a local directory
        if os.path.isdir(model_name_or_path):
            print(f"Loading model from local directory: {model_name_or_path}")
            model_path = model_name_or_path
        else:
            # Download the model using the utility function from faster_translate/utils.py
            model_path = download_model_hf(model_name_or_path, save_path, revision, token)
        
        return TranslateModel(model_path, **kwargs)
        
        
            
            
            
        
            
        

            
            
        


