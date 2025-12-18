# 🇫🇷🇬🇧 English-French Translator Model

![AI Powered](https://img.shields.io/badge/AI-Powered-blueviolet?style=for-the-badge)
![Transformers](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104.0-green?style=for-the-badge)

A robust and efficient Neutral Machine Translation (NMT) model fine‑tuned for translating English text into French. This project leverages the **Helsinki‑NLP/opus‑mt‑en‑fr** pre‑trained model, fine‑tuned on the **Opus Books** dataset, and now ships with a lightweight **FastAPI** backend for easy deployment.

---

## 🚀 Features

- **Fine‑Tuned Precision** – Optimized using the Opus Books dataset for literary‑style translations.
- **State‑of‑the‑Art Architecture** – Built on top of the MarianMT architecture.
- **FastAPI Backend** – Ready‑to‑use REST API for single and batch translations.
- **Easy Integration** – Uses Hugging Face `pipeline` for seamless translation.
- **Evaluation Metrics** – Rigorously evaluated using BLEU scores with `sacrebleu`.

## 🛠️ Technology Stack

- **Python**
- **Hugging Face Transformers**
- **Datasets** (Hugging Face)
- **Evaluate & SacreBLEU**
- **PyTorch / TensorFlow** (backend)
- **FastAPI & Uvicorn** (API server)

## 📦 Installation & Requirements

```bash
# Install all dependencies (including the API server)
pip install -r requirements.txt
```

## 📂 Repository Structure

```
├─ app.py               # FastAPI server exposing the translation model
├─ requirements.txt     # Python dependencies
├─ my_translation_model # Fine‑tuned model files (config, tokenizer, weights)
├─ model.ipynb          # Jupyter notebook used for training
├─ README.md            # This documentation
└─ .gitignore          # Ignores virtual env & caches
```

## ▶️ Running the API Server

```bash
# Activate your virtual environment if you have one
# python -m venv venv && source venv/bin/activate   (Windows: venv\Scripts\activate)

# Start the server
python app.py
```
The server will be available at `http://localhost:8000`. Swagger UI can be accessed at `http://localhost:8000/docs`.

## 📡 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/` | Health check – returns service status and available endpoints |
| `GET`  | `/health` | Detailed health check with model metadata |
| `POST` | `/translate` | Translate a single English sentence to French |
| `POST` | `/translate/batch` | Translate a list of English sentences (max 50 per request) |

### Request / Response examples
#### Single translation
```json
POST /translate
{
  "text": "Hello, how are you?",
  "max_length": 128
}
```
```json
Response:
{
  "original_text": "Hello, how are you?",
  "translated_text": "Bonjour, comment allez‑vous?",
  "source_language": "en",
  "target_language": "fr"
}
```
#### Batch translation
```json
POST /translate/batch
{
  "texts": ["I love reading books.", "Machine learning is fascinating."],
  "max_length": 128
}
```
```json
Response:
{
  "translations": [
    {"original_text": "I love reading books.", "translated_text": "J'aime lire des livres."},
    {"original_text": "Machine learning is fascinating.", "translated_text": "L'apprentissage automatique est fascinant."}
  ]
}
```

## 🧠 Model Training (unchanged)

The training process involves:
1. **Tokenizer** – `MarianTokenizer`
2. **Model** – `MarianMTModel` pre‑trained on English‑French
3. **Pre‑processing** – truncation & padding to a max length of 128 tokens
4. **Fine‑tuning** – `Seq2SeqTrainer` with mixed‑precision (`fp16`)

### Hyper‑parameters (example)
- Learning Rate: `3e-5`
- Batch Size: `8`
- Epochs: `1`
- Weight Decay: `0.01`

## 📊 Evaluation

The model performance is evaluated using the **BLEU** score:
```python
results = trainer.evaluate()
print(f"Final BLEU Score: {results['eval_bleu']:.2f}")
```

## 💾 Saving the Model

The fine‑tuned model is saved to `./my_finetuned_en_fr_translator` and can be zipped for deployment:
```bash
zip -r my_translation_model.zip my_finetuned_en_fr_translator
```

---

**Author**: [Your Name]
**License**: MIT
