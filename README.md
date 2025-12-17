# 🇫🇷🇬🇧 English-French Translator Model

![AI Powered](https://img.shields.io/badge/AI-Powered-blueviolet?style=for-the-badge)
![Transformers](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge)

A robust and efficient Neutral Machine Translation (NMT) model fine-tuned for translating English text into French. This project leverages the power of the **Helsinki-NLP/opus-mt-en-fr** pre-trained model and fine-tunes it on the **Opus Books** dataset to achieve high-quality translations.

---

## 🚀 Features

- **Fine-Tuned Precision**: Optimized using the Opus Books dataset for literary-style translations.
- **State-of-the-Art Architecture**: Built on top of the MarianMT architecture.
- **Easy Integration**: Uses Hugging Face's `pipeline` for seamless translation.
- **Evaluation Metrics**: rigorously evaluated using BLEU scores with `sacrebleu`.

## 🛠️ Technology Stack

- **Python**
- **Hugging Face Transformers**
- **Datasets** (Hugging Face)
- **Evaluate** & **SacreBLEU**
- **PyTorch** / **TensorFlow** (Backend)

## 📦 Installation & Requirements

To run this model locally or in a notebook environment, ensure you have the following dependencies installed:

```bash
pip install transformers datasets evaluate sacrebleu torch
```

## 📖 Dataset

The model is trained on the **[Opus Books](https://huggingface.co/datasets/opus_books)** dataset, specifically the English-to-French (`en-fr`) subset.

- **Source**: English
- **Target**: French
- **Splits**: The dataset is split into training and testing sets (10% for testing) to ensure robust evaluation.

## 🧠 Model Training

The training process involves:
1.  **Tokenizer**: Using `MarianTokenizer` to process inputs.
2.  **Model**: `MarianMTModel` pre-trained on English-French.
3.  **Preprocessing**: Text truncation and padding to a max length of 128 tokens.
4.  **Training**: Fine-tuning using `Seq2SeqTrainer` with `fp16` mixed precision for efficiency.

### Hyperparameters (Example)
- **Learning Rate**: `3e-5`
- **Batch Size**: 8
- **Epochs**: 1
- **Weight Decay**: 0.01

## 💻 Usage

Run the Jupyter Notebook `model.ipynb` to train the model. Once trained, you can use the model for inference as follows:

```python
from transformers import pipeline

# Load the fine-tuned model (assuming it's loaded in memory or saved)
translator = pipeline("translation", model=model, tokenizer=tokenizer)

text = "Machine learning is fascinating."
translation = translator(text, max_length=128)

print(f"English: {text}")
print(f"French: {translation[0]['translation_text']}")
```

**Output:**
```
English: Machine learning is fascinating.
French: L'apprentissage automatique est fascinant.
```

## 📊 Evaluation

The model performance is evaluated using the **BLEU** score, a standard metric for machine translation quality.

```python
results = trainer.evaluate()
print(f"Final BLEU Score: {results['eval_bleu']:.2f}")
```

## 💾 Saving the Model

The trained model is saved to the `./my_finetuned_en_fr_translator` directory and can be zipped for deployment or transfer using `shutil`.

---

**Author**: [Your Name]
**License**: MIT
