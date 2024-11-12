<a href="https://pepy.tech/projects/faster-translate"><img src="https://static.pepy.tech/badge/faster-translate" alt="PyPI Downloads"></a>
<a href="https://pepy.tech/projects/faster-translate"><img src="https://static.pepy.tech/badge/faster-translate/month" alt="PyPI Downloads"></a>
# Faster Translate

Faster Translate is a high-performance translation library built on top of ctranslate2 and designed for fast and efficient translation. It provides an easy-to-use interface for translating text in various languages with support for pre-trained models from Hugging Face's model hub.

## Features

- High-speed translation leveraging ctranslate2
- Support for loading models directly from Hugging Face's model hub
- Save dataset directly to hf after translation.

## Installation

To install Faster Translate, you can use pip:

```
pip install faster-translate
```

## Usage:

Initialize with supported model name:

```
from faster_translate import TranslateModel

model = TranslateModel.from_pretrained("banglanmt_bn2en")
```

Or, `ct2` converted models, local or from huggingfcae hub:

```
from faster_translate import TranslateModel

model = TranslateModel.from_pretrained(
    "sawradip/faster-translate-banglanmt-bn2en-t5",
    normalizer_func = "buetnlpnormalizer"
                                       )

```

You can translate a single sentence:

```
model.translate_single("দেশে বিদেশি ঋণ নিয়ে এখন বেশ আলোচনা হচ্ছে। এই ঋণ পরিশোধের চাপ ধীরে ধীরে বাড়ছে। গত ২০২২-২৩ অর্থবছরে মোট ২৬৭ কোটি ডলারের ঋণ পরিশোধ করতে হয়েছে। আগামী সাত বছরে ঋণ পরিশোধের পরিমাণ বেড়ে দ্বিগুণ হবে বলে মনে করছে অর্থনৈতিক সম্পর্ক বিভাগ (ইআরডি)।")
```

Or a batch:

```
model.translate_batch([
    "দেশে বিদেশি ঋণ নিয়ে এখন বেশ আলোচনা হচ্ছে। এই ঋণ পরিশোধের চাপ ধীরে ধীরে বাড়ছে। গত ২০২২-২৩ অর্থবছরে মোট ২৬৭ কোটি ডলারের ঋণ পরিশোধ করতে হয়েছে। আগামী সাত বছরে ঋণ পরিশোধের পরিমাণ বেড়ে দ্বিগুণ হবে বলে মনে করছে অর্থনৈতিক সম্পর্ক বিভাগ (ইআরডি)।",
    "রাত তিনটার দিকে কাঁচামাল নিয়ে গুলিস্তান থেকে পুরান ঢাকার শ্যামবাজারের আড়তে যাচ্ছিলেন লিটন ব্যাপারী। "
    ])
```

Translating HF dataset directly:
```
model = TranslateModel.from_pretrained(
                                    "banglanmt_en2bn",
                                       )

model.translate_hf_dataset(
    "sawradip/bn-translation-mega-raw-noisy", 
    batch_size=8
)
```
> Features while translating directly from hf dataset
- automatically translate all subsets or any particular subset by specifiying in the `subset_name` parameter
```
model.translate_hf_dataset(
    "sawradip/bn-translation-mega-raw-noisy",
    subset_name=["google"], 
    batch_size=8
)
```

- automatically translate all splits (train, test, validation) or any particular subset by specifiying in the `split` parameter

- automatically translate full dataset or partially specifying `start_idx` and `end_idx` or `translation_size` parameters.
```
model.translate_hf_dataset(
    "sawradip/bn-translation-mega-raw-noisy",
    subset_name="alt",
    batch_size=8, 
    start_idx=2,
    end_idx=50
)
```

```
model.translate_hf_dataset(
    "sawradip/bn-translation-mega-raw-noisy",
    subset_name="alt",
    batch_size=8, 
    translation_size=0.5
)
```

> Push the translated dataset to hf after translation
```
model.translate_hf_dataset(
    "sawradip/bn-translation-mega-raw-noisy",
    subset_name="alt",
    batch_size=8, 
    translation_size=0.5,
    push_to_hub=True,
    token=<pass your hf token>,
    save_repo_name=<name of the dataset repo to save the data>,
)
```

### Currently Supported Models

- [BanglaNMT(BUET)](https://github.com/csebuetnlp/banglanmt) -> (Bangla -> English) - `banglanmt_bn2en`
- [BanglaNMT(BUET)](https://github.com/csebuetnlp/banglanmt) -> (English -> Bangla) - `banglanmt_en2bn`
- `bangla_mbartv1_en2bn`

## 💪 Thanks To All Contributors

<a href="https://github.com/sawradip/faster-translate/graphs/contributors">
  <img src="https://contributors-img.web.app/image?repo=sawradip/faster-translate" alt="List of Contributors"/>
</a>

