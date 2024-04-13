
# Faster Translate

Faster Translate is a high-performance translation library built on top of ctranslate2 and designed for fast and efficient translation. It provides an easy-to-use interface for translating text in various languages with support for pre-trained models from Hugging Face's model hub.

## Features

- High-speed translation leveraging ctranslate2
- Support for loading models directly from Hugging Face's model hub


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

### Currently Supported Models
* [BanglaNMT(BUET)](https://github.com/csebuetnlp/banglanmt) -> (Bangla -> English) - `banglanmt_bn2en`
* [BanglaNMT(BUET)](https://github.com/csebuetnlp/banglanmt) -> (English -> Bangla) - `banglanmt_en2bn`

### Features to be supported:
* Model comversion Scripts
* Direct HF dataset translation





