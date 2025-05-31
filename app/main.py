from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI(title="Tiny GPT2 Next Word Predictor")

# Load tokenizer and model once (small model)
MODEL_NAME = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

class PredictRequest(BaseModel):
    text: str
    max_new_tokens: int = 5  # number of tokens to generate

@app.post("/predict")
async def predict_next_word(req: PredictRequest):
    if not req.text or req.text.strip() == "":
        raise HTTPException(status_code=400, detail="Input text cannot be empty")

    inputs = tokenizer.encode(req.text, return_tensors="pt")

    # Limit input length to avoid memory spike
    if inputs.size(1) > 30:
        inputs = inputs[:, -30:]  # last 30 tokens only

    # Generate output tokens
    with torch.no_grad():
        output = model.generate(
            inputs,
            max_new_tokens=req.max_new_tokens,
            do_sample=False,  # deterministic output
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = output[0][inputs.size(1):]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return {"input": req.text, "predicted": generated_text.strip()}

@app.get("/demo")
async def root():
    return {"message": "Welcome to Tiny GPT2 Next Word Predictor API. Use POST /predict with JSON {text: 'your input'}"}

@app.get("/")
async def root():
    return {"message": "Welcome to Tiny GPT2 Next Word Predictor API. Use POST /predict with JSON {text: 'your input'}"}
