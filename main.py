from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import os # Import os module to get environment variables

# Initialize FastAPI app
app = FastAPI(
    title="GPT-2 Next Word Prediction API",
    description="An API to predict the next word using a pre-trained GPT-2 model.",
    version="1.0.0"
)

# --- Model Loading ---
# Load model and tokenizer globally when the app starts
# This ensures it's loaded once when the service initializes
# and not on every request.
try:
    print("Loading GPT-2 Tokenizer and Model...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval() # Set model to evaluation mode
    print("GPT-2 Tokenizer and Model loaded successfully.")
except Exception as e:
    print(f"Error loading GPT-2 model: {e}")
    # You might want to handle this more gracefully for production,
    # but for a free tier, it often means an issue with the environment
    # or a transient network problem during startup.
    tokenizer = None
    model = None

# --- Pydantic Model for Request Body ---
class PredictionRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 1 # Default to 1 token as per your original function
    do_sample: bool = True
    top_k: int = 50
    top_p: float = 0.95
    temperature: float = 1.0

# --- Prediction Function (modified to use global model/tokenizer) ---
def predict_next_word(
    prompt: str,
    max_new_tokens: int,
    do_sample: bool,
    top_k: int,
    top_p: float,
    temperature: float
) -> str:
    if tokenizer is None or model is None:
        raise RuntimeError("Model and tokenizer are not loaded.")

    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Ensure input_ids are on the correct device if using GPU (though unlikely on free Render)
    # if torch.cuda.is_available():
    #     inputs = {k: v.to('cuda') for k, v in inputs.items()}
    #     model.to('cuda') # Also move model to GPU if not already

    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id
    )
    
    new_token = outputs[0][inputs.input_ids.shape[-1]:]
    return tokenizer.decode(new_token, skip_special_tokens=True).strip()

# --- API Endpoints ---

@app.get("/")
async def root():
    return {"message": "Welcome to the GPT-2 Next Word Prediction API! Use /predict to make predictions."}

@app.post("/predict/")
async def predict_word_api(request: PredictionRequest):
    """
    Predicts the next word(s) based on the given prompt using GPT-2.
    """
    try:
        predicted_text = predict_next_word(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            do_sample=request.do_sample,
            top_k=request.top_k,
            top_p=request.top_p,
            temperature=request.temperature
        )
        return {"prompt": request.prompt, "predicted_text": predicted_text}
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

# --- Health Check (Optional but good practice for Render) ---
# Render often uses a health check to know if your service is ready.
@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify if the API is running and model is loaded.
    """
    if tokenizer is not None and model is not None:
        return {"status": "ok", "model_loaded": True}
    else:
        raise HTTPException(status_code=503, detail="Model not loaded or still loading.")