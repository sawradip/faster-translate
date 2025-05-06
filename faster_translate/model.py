import os
import json
import subprocess
import ctranslate2
from tqdm.auto import tqdm
from datasets import load_dataset, DatasetDict
from tokenizers import Tokenizer
from tokenizers.models import BPE
from transformers import AutoTokenizer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from typing import List, Dict, Union, Optional, Any

# Import vllm conditionally to handle environments where it's not installed
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

from .utils import download_model_hf, _MODELS


def is_cuda_available():
    """Check if CUDA is available on the system."""
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


class TranslatorTokenizer:
    """
    Handles tokenization/detokenization for translation tasks.
    Supports both custom BPE tokenizers and HuggingFace tokenizers.
    """
    
    def __init__(self, model_dir, source_token_list=None, tokenizer_filename=None, tokenizer_repo=None):
        """
        Initialize the tokenizer.
        
        Args:
            model_dir: Directory containing model and tokenizer files
            source_token_list: List of tokens for the BPE tokenizer
            tokenizer_filename: Filename for the vocabulary file
            tokenizer_repo: HuggingFace tokenizer repository
        """
        self.model_dir = model_dir
        self.tokenizer = self._get_tokenizer(tokenizer_repo, source_token_list, tokenizer_filename)
    
    def _get_tokenizer(self, tokenizer_repo, source_token_list, tokenizer_filename):
        """
        Get the appropriate tokenizer based on available resources.
        
        Returns:
            A HuggingFace tokenizer or custom BPE tokenizer
        """
        # Try to load HuggingFace tokenizer first
        hf_tokenizer = self._get_hf_tokenizer(tokenizer_repo)
        if hf_tokenizer is not None:
            return hf_tokenizer
        
        # Fall back to BPE tokenizer
        if source_token_list is None:
            source_token_list = self._get_token_list(tokenizer_filename)
        return self._get_bpe_tokenizer(source_token_list)
    
    def _get_hf_tokenizer(self, tokenizer_repo):
        """Load a HuggingFace tokenizer if available."""
        if tokenizer_repo is not None:
            return AutoTokenizer.from_pretrained(tokenizer_repo)
        elif os.path.isfile(os.path.join(self.model_dir, 'tokenizer_config.json')):
            return AutoTokenizer.from_pretrained(self.model_dir)
        else:
            return None
    
    def _get_token_list(self, tokenizer_filename):
        """Load token list from vocabulary file."""
        if tokenizer_filename is not None:
            tokenizer_filepath = os.path.join(self.model_dir, tokenizer_filename)
        elif os.path.isfile(os.path.join(self.model_dir, "source_vocabulary.json")):
            tokenizer_filepath = os.path.join(self.model_dir, "source_vocabulary.json")
        elif os.path.isfile(os.path.join(self.model_dir, "shared_vocabulary.json")):
            tokenizer_filepath = os.path.join(self.model_dir, "shared_vocabulary.json")
        else:
            raise Exception("Vocabulary file was not found.")
        
        with open(tokenizer_filepath, 'r', encoding='utf-8') as file:
            source_token_list = json.load(file)
        return source_token_list
    
    def _get_bpe_tokenizer(self, token_list):
        """Create a BPE tokenizer with the provided token list."""
        # Initialize a tokenizer
        tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = Whitespace()
        
        # Trainer to train the tokenizer
        trainer = BpeTrainer(special_tokens=token_list, vocab_size=len(token_list), min_frequency=1)
        tokenizer.train_from_iterator(token_list, trainer=trainer)
        
        return tokenizer
    
    def tokenize_batch(self, text_batch):
        """
        Tokenize a batch of texts.
        
        Args:
            text_batch: List of text strings to tokenize
            
        Returns:
            List of token lists for each text
        """
        if isinstance(self.tokenizer, Tokenizer):
            text_batch = [f" {text}".replace(" ", "▁") for text in text_batch]
            tokenized_batch = [encoded.tokens for encoded in self.tokenizer.encode_batch(text_batch)]
        else:
            tokenized_batch = [self.tokenizer.convert_ids_to_tokens(
                input_id_list) for input_id_list in self.tokenizer(text_batch)["input_ids"]]
        return tokenized_batch
    
    def detokenize_single(self, tokens):
        """
        Detokenize a single list of tokens.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Detokenized text
        """
        if isinstance(self.tokenizer, Tokenizer):
            concatenated = ''.join(tokens)
            proper_string = concatenated.replace('▁', ' ').strip()
            proper_string = proper_string.replace(' ,', ',').replace(' .', '.').replace(" '", "'").replace(" :", ":")
        else:
            proper_string = self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(tokens), skip_special_tokens=True)
        return proper_string
    
    def detokenize_batch(self, token_batch):
        """
        Detokenize a batch of token lists.
        
        Args:
            token_batch: Batch of token lists from the translator
            
        Returns:
            List of detokenized texts
        """
        translated_decoded_batch = [self.detokenize_single(pred.hypotheses[0]) for pred in token_batch]
        return translated_decoded_batch


class TextNormalizer:
    """Handles text normalization for different languages."""
    
    @staticmethod
    def get_normalizer(normalizer_name=None):
        """
        Get a text normalizer function based on the normalizer name.
        
        Args:
            normalizer_name: Name of the normalizer or None for identity function
            
        Returns:
            Normalizer function that takes a list of texts and returns normalized texts
        """
        if normalizer_name is None:
            return lambda x: x
        
        if normalizer_name == "buetnlpnormalizer":
            try:
                from normalizer import normalize
            except ImportError:
                raise Exception("Normalizer not installed, please install with `pip install git+https://github.com/csebuetnlp/normalizer`")
            
            return lambda texts: [normalize(text) for text in texts]
        else:
            raise Exception(f"{normalizer_name} not supported yet.")


# class VLLMTranslatorModel:
#     """
#     Translator model that uses vLLM for inference.
#     Specialized for models like Qwen that require a different inference approach.
#     """
    
#     def __init__(self, 
#                  model_name, 
#                  tokenizer_name=None,
#                  max_model_len=4096,
#                  gpu_memory_utilization=0.5,
#                  dtype="bfloat16",
#                  token=None,
#                  sampling_params=None):
#         """
#         Initialize the vLLM translator model.
        
#         Args:
#             model_name: Name of the vLLM model
#             tokenizer_name: Name of the tokenizer (defaults to model_name if None)
#             max_model_len: Maximum model length for inference
#             gpu_memory_utilization: GPU memory utilization (0.0 to 1.0)
#             dtype: Data type for model weights
#             token: HuggingFace token for private models
#             sampling_params: SamplingParams instance or dict for generation
#         """
#         if not VLLM_AVAILABLE:
#             raise ImportError("vLLM is not installed. Please install it with `pip install vllm`")
            
#         self.model_name = model_name
#         self.tokenizer_name = tokenizer_name if tokenizer_name else model_name
        
#         # Initialize tokenizer
#         self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, token=token)
        
#         os.environ["HF_TOKEN"] = token
#         # Initialize LLM
#         self.llm = LLM(
#             model=self.model_name, 
#             max_model_len=max_model_len, 
#             gpu_memory_utilization=gpu_memory_utilization, 
#             dtype=dtype
#         )
        
#         # Set default sampling parameters if not provided
#         if sampling_params is None:
#             self.sampling_params = SamplingParams(
#                 temperature=0.0,
#                 top_p=1.0,
#                 max_tokens=1024
#             )
#         elif isinstance(sampling_params, dict):
#             self.sampling_params = SamplingParams(**sampling_params)
#         else:
#             self.sampling_params = sampling_params
    
#     def translate_single(self, text):
#         """
#         Translate a single text using the vLLM model.
        
#         Args:
#             text: Text to translate
            
#         Returns:
#             Translated text
#         """
#         prompt = self.tokenizer.apply_chat_template([{"role": "user", "content": text}], 
#                                                   tokenize=False, 
#                                                   add_generation_prompt=True)
        
#         outputs = self.llm.generate([prompt], self.sampling_params)
#         return outputs[0].outputs[0].text
    
#     def translate_batch(self, text_batch):
#         """
#         Translate a batch of texts using the vLLM model.
        
#         Args:
#             text_batch: List of texts to translate
            
#         Returns:
#             List of translated texts
#         """
#         prompts = [
#             self.tokenizer.apply_chat_template([{"role": "user", "content": text}], 
#                                              tokenize=False, 
#                                              add_generation_prompt=True)
#             for text in text_batch
#         ]
        
#         outputs = self.llm.generate(prompts, self.sampling_params)
#         return [output.outputs[0].text for output in outputs]
    
#     def translate_bulk(self, text_list, batch_size=16):
#         """
#         Translate a large list of texts in batches with progress bar.
        
#         Args:
#             text_list: List of texts to translate
#             batch_size: Number of texts to translate at once
            
#         Returns:
#             List of translated texts
#         """
#         text_batches = [text_list[i:i + batch_size] for i in range(0, len(text_list), batch_size)]
        
#         translated_results = []
#         for text_batch in tqdm(text_batches):
#             translated_batch = self.translate_batch(text_batch)
#             translated_results.extend(translated_batch)
#         return translated_results
#     def translate_hf_dataset(self, 
#                         dataset_repo, 
#                         subset_name=None,
#                         split=["train"], 
#                         columns=[], 
#                         batch_size=16, 
#                         token=None,
#                         translation_size=None,
#                         start_idx=0,
#                         end_idx=None,
#                         output_format="json",
#                         output_name="preds.json",
#                         push_to_hub=False,
#                         save_repo_name=None,
#                         keep_other_columns=True):
#         """
#         Translate columns from a HuggingFace dataset.
        
#         Args:
#             dataset_repo: HuggingFace dataset repository
#             subset_name: Subset of the dataset
#             split: Train, test, validation split
#             columns: Which columns to translate
#             batch_size: Batch size
#             token: HuggingFace token
#             translation_size: Dataset percentage to translate
#             start_idx: Starting index
#             end_idx: Ending index
#             output_format: Dataset output format
#             output_name: Output file name
#             push_to_hub: Push to hub or not
#             save_repo_name: HuggingFace dataset name to save to
#             keep_other_columns: Whether to keep columns that are not being translated
            
#         Returns:
#             Dictionary containing translated data
#         """
#         dataset_args = [dataset_repo]
#         if subset_name is not None:
#             dataset_args.append(subset_name)
            
#         dataset_kwargs = {"verification_mode": "no_checks" }
#         if token is not None:
#             dataset_kwargs["token"] = token
#         dataset = load_dataset(*dataset_args, **dataset_kwargs)
        
#         if split == "*":
#             split = [split for split in dataset.keys()]
#         if isinstance(split, str):
#             split = [split]
        
#         temp_dataset = {}
#         final_dataset_dict = {}
#         static_end_idx = end_idx
        
#         for split_name in split:
#             split_data = dataset[split_name]
#             flattened_data_list = []
#             data_length_map = []
#             final_dataset_dict[split_name] = {}
            
#             # Handle indices for dataset slicing
#             if translation_size is None:
#                 # Handling last index for full dataset
#                 if static_end_idx is None:
#                     end_idx = len(split_data)
                
#                 # Handling negative indices properly for each split
#                 _start_idx = len(split_data) + start_idx if start_idx < 0 else start_idx
#                 _end_idx = (len(split_data) + end_idx if end_idx < 0 else end_idx)
#             else:
#                 _start_idx = len(split_data) + start_idx if start_idx < 0 else start_idx
#                 _end_idx = int(len(split_data) * translation_size) if translation_size <= 1 else translation_size
            
#             # Initialize with selected dataset slice
#             temp_dataset[split_name] = split_data.select(range(_start_idx, _end_idx))
            
#             # Include original columns in final_dataset_dict if keeping other columns
#             if keep_other_columns:
#                 for original_col in split_data.column_names:
#                     if original_col not in columns:
#                         final_dataset_dict[split_name][original_col] = temp_dataset[split_name][original_col]
            
#             for column in columns:
#                 print(f"\033[34mTranslating {split_name} split from {_start_idx} to {_end_idx} of column {column}.")
#                 data_list = split_data[column]
                
#                 # Handle different data formats (list of strings, nested lists)
#                 if isinstance(data_list[0], list) or (
#                     isinstance(data_list[0], str) and 
#                     data_list[0].startswith("[") and 
#                     data_list[0].endswith("]") and 
#                     isinstance(eval(data_list[0]), list)
#                 ):
#                     for sample in data_list[_start_idx:_end_idx]:
#                         sample = sample if isinstance(data_list[0], list) else eval(sample)
#                         data_length_map.append(len(sample))
#                         flattened_data_list.extend(sample)
#                 elif isinstance(data_list[0], str):
#                     flattened_data_list = data_list[_start_idx:_end_idx]
#                 else:
#                     raise Exception(f"We only support of `str` or `List[str]` type columns, not `{type(data_list[0])}`")
                
#                 # Translate the flattened data
#                 translated_flattened_data_list = self.translate_bulk(flattened_data_list, batch_size=batch_size)
                
#                 # Reconstruct the original data structure
#                 index_ptr = 0
#                 translated_data_list = []
#                 if not data_length_map:
#                     translated_data_list = translated_flattened_data_list
#                 else:
#                     for data_length in data_length_map:
#                         translated_data_list.append(translated_flattened_data_list[index_ptr: index_ptr + data_length])
#                         index_ptr += data_length
                
#                 # Store the translated content with the original column name (replace original)
#                 final_dataset_dict[split_name][column] = translated_data_list
                
#                 # Replace original column with translated content in the dataset
#                 if len(temp_dataset[split_name]) == len(translated_data_list):
#                     # First remove the original column
#                     temp_dataset[split_name] = temp_dataset[split_name].remove_columns([column])
#                     # Then add the translated content with the original column name
#                     temp_dataset[split_name] = temp_dataset[split_name].add_column(column, translated_data_list)
#                 else: 
#                     print("Given data and Translated data length doesn't match")
#                     print(f"Length of given dataset: {len(temp_dataset[split_name])}", f"Length of translated Data: {len(translated_data_list)}", sep='\n')
        
#         # Save the data
#         with open(output_name, 'w', encoding='utf-8') as f:
#             json.dump(final_dataset_dict, f, ensure_ascii=False, indent=4)
        
#         # Push to HuggingFace Hub if requested
#         if push_to_hub:
#             if save_repo_name is not None:
#                 temp_dataset = DatasetDict(temp_dataset)
#                 temp_dataset.push_to_hub(repo_id=save_repo_name, token=token)
#             else:
#                 print("Please provide a valid huggingface repo name for saving the dataset.")
            
#         return final_dataset_dict
# Modifications to enable streaming in vLLM and dataset loading

class VLLMTranslatorModel:
    """
    Translator model that uses vLLM for inference with streaming support.
    Specialized for models like Qwen that require a different inference approach.
    """
    
    def __init__(self, 
                 model_name, 
                 tokenizer_name=None,
                 max_model_len=4096,
                 gpu_memory_utilization=0.5,
                 dtype="bfloat16",
                 token=None,
                 sampling_params=None,
                 enable_streaming=False):
        """
        Initialize the vLLM translator model with optional streaming support.
        
        Args:
            model_name: Name of the vLLM model
            tokenizer_name: Name of the tokenizer (defaults to model_name if None)
            max_model_len: Maximum model length for inference
            gpu_memory_utilization: GPU memory utilization (0.0 to 1.0)
            dtype: Data type for model weights
            token: HuggingFace token for private models
            sampling_params: SamplingParams instance or dict for generation
            enable_streaming: Whether to enable token streaming during generation
        """
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM is not installed. Please install it with `pip install vllm`")
            
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name if tokenizer_name else model_name
        self.enable_streaming = enable_streaming
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, token=token)
        
        os.environ["HF_TOKEN"] = token
        # Initialize LLM
        self.llm = LLM(
            model=self.model_name, 
            max_model_len=max_model_len, 
            gpu_memory_utilization=gpu_memory_utilization, 
            dtype=dtype
        )
        
        # Set default sampling parameters if not provided
        if sampling_params is None:
            self.sampling_params = SamplingParams(
                temperature=0.3,
                top_p=1.0,
                max_tokens=1024
            )
        elif isinstance(sampling_params, dict):
            self.sampling_params = SamplingParams(**sampling_params)
        else:
            self.sampling_params = sampling_params
    def translate_bulk(self, text_list, batch_size=16):
        """
        Translate a large list of texts in batches with progress bar.
        
        Args:
            text_list: List of texts to translate
            batch_size: Number of texts to translate at once
            
        Returns:
            List of translated texts
        """
        text_batches = [text_list[i:i + batch_size] for i in range(0, len(text_list), batch_size)]
        
        translated_results = []
        for text_batch in tqdm(text_batches):
            translated_batch = self.translate_batch(text_batch)
            translated_results.extend(translated_batch)
        return translated_results
    
    def translate_single(self, text, callback=None):
        """
        Translate a single text using the vLLM model, with optional streaming.
        
        Args:
            text: Text to translate
            callback: Optional callback function for streaming mode
            
        Returns:
            Translated text or generator if streaming
        """
        prompt = self.tokenizer.apply_chat_template([{"role": "user", "content": text}], 
                                                  tokenize=False, 
                                                  add_generation_prompt=True)
        
        if self.enable_streaming and callback is not None:
            return self._stream_translation(prompt, callback)
        else:
            outputs = self.llm.generate([prompt], self.sampling_params)
            return outputs[0].outputs[0].text
    
    def _stream_translation(self, prompt, callback):
        """
        Stream translation results token by token.
        
        Args:
            prompt: Prompt to generate from
            callback: Function to call with each new token
            
        Returns:
            Generator yielding tokens as they are generated
        """
        # Set streaming in sampling params
        streaming_params = SamplingParams(
            **self.sampling_params.__dict__,
            stream=True
        )
        
        # Get the request ID
        request_id = self.llm.generate([prompt], streaming_params)[0].request_id
        
        # Return a generator that yields each new token
        output_text = ""
        for output in self.llm.stream_results([request_id]):
            if output.outputs:
                next_token = output.outputs[0].text[len(output_text):]
                output_text = output.outputs[0].text
                if next_token:
                    callback(next_token)
                    yield next_token
    
    def translate_batch(self, text_batch):
        """
        Translate a batch of texts using the vLLM model.
        Streaming is not supported for batch translation.
        
        Args:
            text_batch: List of texts to translate
            
        Returns:
            List of translated texts
        """
        if self.enable_streaming:
            print("Warning: Streaming mode is not supported for batch translation.")
        
        prompts = [
            self.tokenizer.apply_chat_template([{"role": "user", "content": text}], 
                                             tokenize=False, 
                                             add_generation_prompt=True)
            for text in text_batch
        ]
        
        outputs = self.llm.generate(prompts, self.sampling_params)
        return [output.outputs[0].text for output in outputs]
    
    def translate_hf_dataset(self, 
                        dataset_repo, 
                        subset_name=None,
                        split=["train"], 
                        columns=[], 
                        batch_size=16, 
                        token=None,
                        translation_size=None,
                        start_idx=0,
                        end_idx=None,
                        output_format="json",
                        output_name="preds.json",
                        push_to_hub=False,
                        save_repo_name=None,
                        keep_other_columns=True,
                        streaming_dataset=False):
        """
        Translate columns from a HuggingFace dataset with optional streaming.
        
        Args:
            dataset_repo: HuggingFace dataset repository
            subset_name: Subset of the dataset
            split: Train, test, validation split
            columns: Which columns to translate
            batch_size: Batch size
            token: HuggingFace token
            translation_size: Dataset percentage to translate
            start_idx: Starting index
            end_idx: Ending index
            output_format: Dataset output format
            output_name: Output file name
            push_to_hub: Push to hub or not
            save_repo_name: HuggingFace dataset name to save to
            keep_other_columns: Whether to keep columns that are not being translated
            streaming_dataset: Whether to stream the dataset instead of loading it all at once
            
        Returns:
            Dictionary containing translated data or a generator if streaming
        """
        dataset_args = [dataset_repo]
        if subset_name is not None:
            dataset_args.append(subset_name)
            
        dataset_kwargs = {"verification_mode": "no_checks"}
        if token is not None:
            dataset_kwargs["token"] = token
        
        # Add streaming parameter
        if streaming_dataset:
            dataset_kwargs["streaming"] = True
        
        dataset = load_dataset(*dataset_args, **dataset_kwargs)
        
        if split == "*":
            split = [split for split in dataset.keys()]
        if isinstance(split, str):
            split = [split]
        
        # Handle streaming dataset
        if streaming_dataset:
            return self._translate_streaming_dataset(
                dataset, split, columns, batch_size, 
                translation_size, start_idx, end_idx,
                output_format, output_name
            )
        
        # Original non-streaming implementation
        temp_dataset = {}
        final_dataset_dict = {}
        static_end_idx = end_idx
        
        for split_name in split:
            split_data = dataset[split_name]
            flattened_data_list = []
            data_length_map = []
            final_dataset_dict[split_name] = {}
            
            # Handle indices for dataset slicing
            if translation_size is None:
                # Handling last index for full dataset
                if static_end_idx is None:
                    end_idx = len(split_data)
                
                # Handling negative indices properly for each split
                _start_idx = len(split_data) + start_idx if start_idx < 0 else start_idx
                _end_idx = (len(split_data) + end_idx if end_idx < 0 else end_idx)
            else:
                _start_idx = len(split_data) + start_idx if start_idx < 0 else start_idx
                _end_idx = int(len(split_data) * translation_size) if translation_size <= 1 else translation_size
            
            # Initialize with selected dataset slice
            temp_dataset[split_name] = split_data.select(range(_start_idx, _end_idx))
            
            # Include original columns in final_dataset_dict if keeping other columns
            if keep_other_columns:
                for original_col in split_data.column_names:
                    if original_col not in columns:
                        final_dataset_dict[split_name][original_col] = temp_dataset[split_name][original_col]
            
            for column in columns:
                print(f"\033[34mTranslating {split_name} split from {_start_idx} to {_end_idx} of column {column}.")
                data_list = split_data[column]
                
                # Handle different data formats (list of strings, nested lists)
                if isinstance(data_list[0], list) or (
                    isinstance(data_list[0], str) and 
                    data_list[0].startswith("[") and 
                    data_list[0].endswith("]") and 
                    isinstance(eval(data_list[0]), list)
                ):
                    for sample in data_list[_start_idx:_end_idx]:
                        sample = sample if isinstance(data_list[0], list) else eval(sample)
                        data_length_map.append(len(sample))
                        flattened_data_list.extend(sample)
                elif isinstance(data_list[0], str):
                    flattened_data_list = data_list[_start_idx:_end_idx]
                else:
                    raise Exception(f"We only support of `str` or `List[str]` type columns, not `{type(data_list[0])}`")
                
                # Translate the flattened data
                translated_flattened_data_list = self.translate_bulk(flattened_data_list, batch_size=batch_size)
                
                # Reconstruct the original data structure
                index_ptr = 0
                translated_data_list = []
                if not data_length_map:
                    translated_data_list = translated_flattened_data_list
                else:
                    for data_length in data_length_map:
                        translated_data_list.append(translated_flattened_data_list[index_ptr: index_ptr + data_length])
                        index_ptr += data_length
                
                # Store the translated content with the original column name (replace original)
                final_dataset_dict[split_name][column] = translated_data_list
                
                # Replace original column with translated content in the dataset
                if len(temp_dataset[split_name]) == len(translated_data_list):
                    # First remove the original column
                    temp_dataset[split_name] = temp_dataset[split_name].remove_columns([column])
                    # Then add the translated content with the original column name
                    temp_dataset[split_name] = temp_dataset[split_name].add_column(column, translated_data_list)
                else: 
                    print("Given data and Translated data length doesn't match")
                    print(f"Length of given dataset: {len(temp_dataset[split_name])}", f"Length of translated Data: {len(translated_data_list)}", sep='\n')
        
        # Save the data
        with open(output_name, 'w', encoding='utf-8') as f:
            json.dump(final_dataset_dict, f, ensure_ascii=False, indent=4)
        
        # Push to HuggingFace Hub if requested
        if push_to_hub:
            if save_repo_name is not None:
                temp_dataset = DatasetDict(temp_dataset)
                temp_dataset.push_to_hub(repo_id=save_repo_name, token=token)
            else:
                print("Please provide a valid huggingface repo name for saving the dataset.")
            
        return final_dataset_dict
    
    def _translate_streaming_dataset(self,
                                   dataset,
                                   split,
                                   columns,
                                   batch_size,
                                   translation_size,
                                   start_idx,
                                   end_idx,
                                   output_format,
                                   output_name):
        """
        Process a streaming dataset, translating it in chunks.
        
        This method yields translated samples as they are processed,
        avoiding loading the entire dataset into memory.
        """
        for split_name in split:
            stream = dataset[split_name]
            
            # Apply skip and take for streaming datasets
            if start_idx > 0:
                stream = stream.skip(start_idx)
            
            # Limit the number of items to process
            if translation_size is not None:
                if translation_size <= 1.0:
                    # We can't calculate percentage for streaming, so warn user
                    print("Warning: Percentage-based translation_size not supported for streaming. Using as absolute value.")
                stream = stream.take(int(translation_size))
            elif end_idx is not None:
                items_to_take = end_idx - start_idx
                stream = stream.take(items_to_take)
            
            # Process in mini-batches to balance memory usage and efficiency
            current_batch = []
            processed_count = 0
            
            # Save output as we go
            if output_format == "json":
                output_file = open(output_name, 'w', encoding='utf-8')
                output_file.write('{"results": [')
                first_item = True
            
            try:
                for sample in tqdm(stream):
                    processed_samples = {}
                    
                    # Process each column that needs translation
                    for column in columns:
                        if column in sample:
                            # Handle text and list formats
                            data = sample[column]
                            if isinstance(data, list) or (
                                isinstance(data, str) and 
                                data.startswith("[") and 
                                data.endswith("]") and 
                                isinstance(eval(data), list)
                            ):
                                data_list = data if isinstance(data, list) else eval(data)
                                translated_items = []
                                for item in data_list:
                                    translated_items.append(self.translate_single(item))
                                processed_samples[column] = translated_items
                            else:
                                processed_samples[column] = self.translate_single(data)
                    
                    # Add all other columns
                    for key, value in sample.items():
                        if key not in columns:
                            processed_samples[key] = value
                    
                    # Write to output file
                    if output_format == "json":
                        if not first_item:
                            output_file.write(',')
                        else:
                            first_item = False
                        output_file.write(json.dumps(processed_samples, ensure_ascii=False))
                    
                    processed_count += 1
                    yield processed_samples
            
            finally:
                # Close the output file
                if output_format == "json":
                    output_file.write(']}')
                    output_file.close()
                    
            print(f"Processed {processed_count} items from {split_name} split")

class TranslatorModel:
    """
    Main translator class that handles the translation process.
    Uses a tokenizer and a translator engine to translate text.
    """
    
    def __init__(self, 
                 model_dir, 
                 tokenizer=None, 
                 source_token_list=None, 
                 tokenizer_filename=None, 
                 tokenizer_repo=None, 
                 normalizer_func=None, 
                 device=device_name):
        """
        Initialize the translator model.
        
        Args:
            model_dir: Directory containing the model files
            tokenizer: Existing tokenizer instance or None to create a new one
            source_token_list: List of tokens for the tokenizer
            tokenizer_filename: Filename for the vocabulary file
            tokenizer_repo: HuggingFace tokenizer repository
            normalizer_func: Function to normalize text before tokenization
            device: Device to run the model on ("cuda" or "cpu")
        """
        self.model_dir = model_dir
        self.translator = ctranslate2.Translator(model_dir, device=device)
        
        # Setup the tokenizer
        if tokenizer is None:
            self.tokenizer = TranslatorTokenizer(
                model_dir, 
                source_token_list=source_token_list,
                tokenizer_filename=tokenizer_filename,
                tokenizer_repo=tokenizer_repo
            )
        else:
            self.tokenizer = tokenizer
        
        # Setup the normalizer
        if isinstance(normalizer_func, str):
            self.normalizer = TextNormalizer.get_normalizer(normalizer_func)
        elif normalizer_func is None:
            self.normalizer = TextNormalizer.get_normalizer(None)
        else:
            self.normalizer = normalizer_func
    
    def translate_single(self, text):
        """
        Translate a single text.
        
        Args:
            text: Text to translate
            
        Returns:
            Translated text
        """
        text_batch = [text]
        translated_batch = self.translate_batch(text_batch)
        return translated_batch[0]
    
    def translate_batch(self, text_batch):
        """
        Translate a batch of texts.
        
        Args:
            text_batch: List of texts to translate
            
        Returns:
            List of translated texts
        """
        normalized_text_batch = self.normalizer(text_batch)
        tokenized_text_batch = self.tokenizer.tokenize_batch(normalized_text_batch)
        translated_tokens_batch = self.translator.translate_batch(tokenized_text_batch)
        translated_batch = self.tokenizer.detokenize_batch(translated_tokens_batch)
        return translated_batch
    
    def translate_bulk(self, text_list, batch_size=16):
        """
        Translate a large list of texts in batches with progress bar.
        
        Args:
            text_list: List of texts to translate
            batch_size: Number of texts to translate at once
            
        Returns:
            List of translated texts
        """
        text_batches = [text_list[i:i + batch_size] for i in range(0, len(text_list), batch_size)]
        
        translated_results = []
        for text_batch in tqdm(text_batches):
            translated_batch = self.translate_batch(text_batch)
            translated_results.extend(translated_batch)
        return translated_results
    
    def translate_file(self, file_path, batch_size, output_file="preds.txt"):
        """
        Translate texts from a file.
        
        Args:
            file_path: Path to the file containing texts to translate
            batch_size: Number of texts to translate at once
            output_file: Path to save the translated texts
            
        Returns:
            List of translated texts
        """
        with open(file_path, 'r') as file:
            text_list = [line.strip() for line in file.readlines()]
            
        translated_results = self.translate_bulk(text_list, batch_size)
        
        if output_file is not None:
            translated_str = "\n".join(translated_results)
            with open(output_file, 'w') as file:
                file.write(translated_str)
                
        return translated_results
    
    def translate_hf_dataset(self, 
                            dataset_repo, 
                            subset_name=None,
                            split=["train"], 
                            columns=[], 
                            batch_size=16, 
                            token=None,
                            translation_size=None,
                            start_idx=0,
                            end_idx=None,
                            output_format="json",
                            output_name="preds.json",
                            push_to_hub=False,
                            save_repo_name=None):
        """
        Translate columns from a HuggingFace dataset.
        
        Args:
            dataset_repo: HuggingFace dataset repository
            subset_name: Subset of the dataset
            split: Train, test, validation split
            columns: Which columns to translate
            batch_size: Batch size
            token: HuggingFace token
            translation_size: Dataset percentage to translate
            start_idx: Starting index
            end_idx: Ending index
            output_format: Dataset output format
            output_name: Output file name
            push_to_hub: Push to hub or not
            save_repo_name: HuggingFace dataset name to save to
            
        Returns:
            Dictionary containing translated data
        """
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
        
        temp_dataset = {}
        final_dataset_dict = {}
        static_end_idx = end_idx
        
        for split_name in split:
            split_data = dataset[split_name]
            flattened_data_list = []
            data_length_map = []
            final_dataset_dict[split_name] = {}
            
            # Handle indices for dataset slicing
            if translation_size is None:
                # Handling last index for full dataset
                if static_end_idx is None:
                    end_idx = len(split_data)
                
                # Handling negative indices properly for each split
                _start_idx = len(split_data) + start_idx if start_idx < 0 else start_idx
                _end_idx = (len(split_data) + end_idx if end_idx < 0 else end_idx)
            else:
                _start_idx = len(split_data) + start_idx if start_idx < 0 else start_idx
                _end_idx = int(len(split_data) * translation_size) if translation_size <= 1 else translation_size
            
            temp_dataset[split_name] = split_data.select(range(_start_idx, _end_idx)) 
            
            for column in columns:
                print(f"\033[34mTranslating {split_name} split from {_start_idx} to {_end_idx} of column {column}.")
                data_list = split_data[column]
                
                # Handle different data formats (list of strings, nested lists)
                if isinstance(data_list[0], list) or (
                    isinstance(data_list[0], str) and 
                    data_list[0].startswith("[") and 
                    data_list[0].endswith("]") and 
                    isinstance(eval(data_list[0]), list)
                ):
                    for sample in data_list[_start_idx:_end_idx]:
                        sample = sample if isinstance(data_list[0], list) else eval(sample)
                        data_length_map.append(len(sample))
                        flattened_data_list.extend(sample)
                elif isinstance(data_list[0], str):
                    flattened_data_list = data_list[_start_idx:_end_idx]
                else:
                    raise Exception(f"We only support of `str` or `List[str]` type columns, not `{type(data_list[0])}`")
                
                # Translate the flattened data
                translated_flattened_data_list = self.translate_bulk(flattened_data_list, batch_size=batch_size)
                
                # Reconstruct the original data structure
                index_ptr = 0
                translated_data_list = []
                if not data_length_map:
                    translated_data_list = translated_flattened_data_list
                else:
                    for data_length in data_length_map:
                        translated_data_list.append(translated_flattened_data_list[index_ptr: index_ptr + data_length])
                        index_ptr += data_length
                
                final_dataset_dict[split_name][column] = translated_data_list
                
                # Add translated column to the dataset
                if len(temp_dataset[split_name]) == len(translated_data_list):
                    temp_dataset[split_name] = temp_dataset[split_name].add_column(f"translated_{column}", translated_data_list)
                else: 
                    print("Given data and Translated data length doesn't match")
                    print(f"Length of given dataset: {len(temp_dataset[split_name])}", f"Length of translated Data: {len(translated_data_list)}", sep='\n')
        
        # Save the data
        with open(output_name, 'w', encoding='utf-8') as f:
            json.dump(final_dataset_dict, f, ensure_ascii=False, indent=4)
        
        # Push to HuggingFace Hub if requested
        if push_to_hub:
            if save_repo_name is not None:
                temp_dataset = DatasetDict(temp_dataset)
                temp_dataset.push_to_hub(repo_id=save_repo_name, token=token)
            else:
                print("Please provide a valid huggingface repo name for saving the dataset.")
            
        return final_dataset_dict
    
    @classmethod
    def from_pretrained(cls, model_name_or_path, save_path=None, revision=None, token=None, tokenizer_repo=None, **kwargs):
        """
        Load a pretrained translation model from HuggingFace or local directory.
        
        Args:
            model_name_or_path: Model name in _MODELS or path to local directory
            save_path: Path to save the downloaded model
            revision: Model revision to download
            token: HuggingFace token
            tokenizer_repo: HuggingFace tokenizer repository
            **kwargs: Additional arguments for TranslatorModel
            
        Returns:
            TranslatorModel or VLLMTranslatorModel instance
        """
        # Check if model is in predefined models
        if model_name_or_path in _MODELS:
            model_args = _MODELS[model_name_or_path]
            
            # Check if this is a vLLM model
            if model_args.get("model_type") == "vllm":
                if not VLLM_AVAILABLE:
                    raise ImportError("vLLM is not installed. Please install it with `pip install vllm`")
                
                # Extract vLLM specific parameters
                vllm_params = {
                    "model_name": model_args.get("model_repo"),
                    "tokenizer_name": model_args.get("tokenizer_repo"),
                    "max_model_len": model_args.get("max_model_len", 4096),
                    "gpu_memory_utilization": model_args.get("gpu_memory_utilization", 0.5),
                    "dtype": model_args.get("dtype", "bfloat16"),
                    "token": token,
                }
                
                # Add any additional parameters from kwargs
                vllm_params.update({k: v for k, v in kwargs.items() if k in [
                    "max_model_len", "gpu_memory_utilization", "dtype", "sampling_params"
                ]})
                
                return VLLMTranslatorModel(**vllm_params)
            
            # For standard CT2 models
            kwargs["normalizer_func"] = model_args.get("normalizer_func")
            kwargs["tokenizer_repo"] = tokenizer_repo if tokenizer_repo else model_args.get("tokenizer_repo")
        
        # Check if model is a local directory
        if os.path.isdir(model_name_or_path):
            print(f"Loading model from local directory: {model_name_or_path}")
            model_path = model_name_or_path
        else:
            # Download the model from HuggingFace
            model_path = download_model_hf(model_name_or_path, save_path, revision, token)
        
        return cls(model_path, **kwargs)