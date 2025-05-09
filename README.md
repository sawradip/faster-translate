# Faster Translate

<div align="center">

[![PyPI Downloads](https://static.pepy.tech/badge/faster-translate)](https://pepy.tech/projects/faster-translate)
[![Monthly Downloads](https://static.pepy.tech/badge/faster-translate/month)](https://pepy.tech/projects/faster-translate)
[![GitHub License](https://img.shields.io/github/license/sawradip/faster-translate)](https://github.com/sawradip/faster-translate/blob/main/LICENSE)
[![PyPI Version](https://img.shields.io/pypi/v/faster-translate)](https://pypi.org/project/faster-translate/)

</div>

A high-performance translation library powered by state-of-the-art models. Faster Translate offers optimized inference using CTranslate2 and vLLM backends, providing an easy-to-use interface for applications requiring efficient and accurate translations.

## 🚀 Features

- **High-performance inference** using CTranslate2 and vLLM backends
- **Seamless integration** with Hugging Face models
- **Flexible API** for single sentence, batch, and large-scale translation
- **Dataset translation** with direct Hugging Face integration
- **Multi-backend support** for both traditional (CTranslate2) and LLM-based (vLLM) models
- **Text normalization** for improved translation quality

## 📦 Installation

### Basic Installation

```bash
pip install faster-translate
```

### With vLLM Support (Recommended)

```bash
pip install faster-translate[vllm]
```

### All Features

```bash
pip install faster-translate[all]
```

## 🔍 Usage

### Basic Translation

```python
from faster_translate import TranslatorModel

# Initialize with a pre-configured model
translator = TranslatorModel.from_pretrained("banglanmt_bn2en")

# Translate a single sentence
english_text = translator.translate_single("দেশে বিদেশি ঋণ নিয়ে এখন বেশ আলোচনা হচ্ছে।")
print(english_text)

# Translate a batch of sentences
bengali_sentences = [
    "দেশে বিদেশি ঋণ নিয়ে এখন বেশ আলোচনা হচ্ছে।",
    "রাত তিনটার দিকে কাঁচামাল নিয়ে গুলিস্তান থেকে পুরান ঢাকার শ্যামবাজারের আড়তে যাচ্ছিলেন লিটন ব্যাপারী।"
]
translations = translator.translate_batch(bengali_sentences)
```

### Using Different Model Backends

```python
# Using a CTTranslate2-based model
ct2_translator = TranslatorModel.from_pretrained("banglanmt_bn2en")

# Using a vLLM-based model
vllm_translator = TranslatorModel.from_pretrained("bangla_qwen_en2bn")
```

### Loading Models from Hugging Face

```python
# Load a specific model from Hugging Face
translator = TranslatorModel.from_pretrained(
    "sawradip/faster-translate-banglanmt-bn2en-t5",
    normalizer_func="buetnlpnormalizer"
)
```

### Translating Hugging Face Datasets

Translate an entire dataset with a single function call:

```python
translator = TranslatorModel.from_pretrained("banglanmt_en2bn")

# Translate the entire dataset
translator.translate_hf_dataset(
    "sawradip/bn-translation-mega-raw-noisy", 
    batch_size=16
)

# Translate specific subsets
translator.translate_hf_dataset(
    "sawradip/bn-translation-mega-raw-noisy",
    subset_name=["google"], 
    batch_size=16
)

# Translate a portion of the dataset
translator.translate_hf_dataset(
    "sawradip/bn-translation-mega-raw-noisy",
    subset_name="alt",
    batch_size=16, 
    translation_size=0.5  # Translate 50% of the dataset
)
```

### Publishing Translated Datasets

Push translated datasets directly to Hugging Face:

```python
translator.translate_hf_dataset(
    "sawradip/bn-translation-mega-raw-noisy",
    subset_name="alt",
    batch_size=16, 
    push_to_hub=True,
    token="your_huggingface_token",
    save_repo_name="your-username/translated-dataset"
)
```

### Enhanced vLLM HuggingFace Dataset Support

When working with challenging datasets, you can use the `verification_mode="no_checks"` parameter:

```python
# Using vLLM backend with relaxed dataset verification
vllm_translator = TranslatorModel.from_pretrained("bangla_qwen_en2bn")

# Translate dataset with relaxed verification checks
vllm_translator.translate_hf_dataset(
    "difficult/dataset-with-formatting-issues",
    batch_size=16,
    # The verification_mode parameter is automatically set to "no_checks"
    # to handle datasets with formatting inconsistencies
)
```
### Advanced Dataset Translation Options

`translate_hf_dataset` supports numerous parameters for fine-grained control:

```python
# Full parameter example
translator.translate_hf_dataset(
    dataset_repo="example/dataset",           # HuggingFace dataset repository
    subset_name="subset",                     # Optional dataset subset
    split=["train", "validation"],            # Dataset splits to translate (default: ["train"])
    columns=["text", "instructions"],         # Columns to translate
    batch_size=32,                            # Number of texts per batch
    token="hf_token",                         # HuggingFace token for private datasets
    translation_size=0.7,                     # Translate 70% of the dataset
    start_idx=100,                            # Start from the 100th example
    end_idx=1000,                             # End at the 1000th example
    output_format="json",                     # Output format
    output_name="translations.json",          # Output file name
    push_to_hub=True,                         # Push translated dataset to HF Hub
    save_repo_name="username/translated-data" # Repository name for upload
)

## 🌐 Supported Models

| Model ID | Source Language | Target Language | Backend | Description |
|----------|----------------|----------------|---------|-------------|
| `banglanmt_bn2en` | Bengali | English | CTranslate2 | BanglaNMT model from BUET |
| `banglanmt_en2bn` | English | Bengali | CTranslate2 | BanglaNMT model from BUET |
| `bangla_mbartv1_en2bn` | English | Bengali | CTranslate2 | MBart-based translation model |
| `bangla_qwen_en2bn` | English | Bengali | vLLM | Qwen-based translation model |

## 🛠️ Advanced Configuration

### Custom Sampling Parameters for vLLM Models

```python
from vllm import SamplingParams

# Create custom sampling parameters
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=512
)

# Initialize translator with custom parameters
translator = TranslatorModel.from_pretrained(
    "bangla_qwen_en2bn", 
    sampling_params=sampling_params
)
```

## 💪 Contributors

<a href="https://github.com/sawradip/faster-translate/graphs/contributors">
  <img src="https://contributors-img.web.app/image?repo=sawradip/faster-translate" alt="List of Contributors"/>
</a>

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 Citation

If you use Faster Translate in your research, please cite:

```bibtex
@software{faster_translate,
  author = {Sawradip Saha and Contributors},
  title = {Faster Translate: High-Performance Machine Translation Library},
  url = {https://github.com/sawradip/faster-translate},
  year = {2024},
}
```