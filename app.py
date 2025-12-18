"""
English-French Translation API
A FastAPI backend to serve the fine-tuned MarianMT translation model.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import MarianMTModel, MarianTokenizer, pipeline
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="English-French Translator API",
    description="A REST API for translating English text to French using a fine-tuned MarianMT model.",
    version="1.0.0"
)

# Add CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and tokenizer on startup
MODEL_PATH = "./my_translation_model"

print("🔄 Loading translation model...")
tokenizer = MarianTokenizer.from_pretrained(MODEL_PATH)
model = MarianMTModel.from_pretrained(MODEL_PATH)
translator = pipeline("translation", model=model, tokenizer=tokenizer)
print("✅ Model loaded successfully!")


# Request/Response models
class TranslationRequest(BaseModel):
    text: str
    max_length: int = 128

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Hello, how are you?",
                "max_length": 128
            }
        }


class TranslationResponse(BaseModel):
    original_text: str
    translated_text: str
    source_language: str = "en"
    target_language: str = "fr"


class BatchTranslationRequest(BaseModel):
    texts: list[str]
    max_length: int = 128


class BatchTranslationResponse(BaseModel):
    translations: list[TranslationResponse]


# API Endpoints
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "message": "English-French Translation API is running!",
        "endpoints": {
            "translate": "/translate",
            "batch_translate": "/translate/batch",
            "docs": "/docs"
        }
    }


@app.post("/translate", response_model=TranslationResponse)
async def translate_text(request: TranslationRequest):
    """
    Translate a single English text to French.
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        result = translator(request.text, max_length=request.max_length)
        translated = result[0]["translation_text"]
        
        return TranslationResponse(
            original_text=request.text,
            translated_text=translated
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")


@app.post("/translate/batch", response_model=BatchTranslationResponse)
async def batch_translate(request: BatchTranslationRequest):
    """
    Translate multiple English texts to French in a single request.
    """
    if not request.texts:
        raise HTTPException(status_code=400, detail="Texts list cannot be empty")
    
    if len(request.texts) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 texts per batch")
    
    try:
        translations = []
        for text in request.texts:
            if text.strip():
                result = translator(text, max_length=request.max_length)
                translations.append(
                    TranslationResponse(
                        original_text=text,
                        translated_text=result[0]["translation_text"]
                    )
                )
        
        return BatchTranslationResponse(translations=translations)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch translation failed: {str(e)}")


@app.get("/health")
async def health_check():
    """Detailed health check with model info"""
    return {
        "status": "healthy",
        "model": {
            "path": MODEL_PATH,
            "type": "MarianMT",
            "source_language": "English",
            "target_language": "French"
        }
    }


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
