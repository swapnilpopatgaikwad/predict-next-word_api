from fastapi import FastAPI
from pydantic import BaseModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Load lightweight model
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
model = GPT2LMHeadModel.from_pretrained("distilgpt2")

# Set to eval mode to disable training behaviors
model.eval()

app = FastAPI(title="Predict Next Word API")


@app.get("/")
def read_root():
    return {"message": "Welcome to Predict Next Word API"}

@app.get("/demo")
def read_root():
    return {"message": "Welcome to Predict Next Word API"}


class InputText(BaseModel):
    text: str

@app.post("/predict")
def predict_next_word(input: InputText):
    inputs = tokenizer.encode(input.text, return_tensors="pt")
    outputs = model.generate(inputs, max_length=inputs.shape[1] + 1, do_sample=True)
    next_token_id = outputs[0][-1].item()
    next_word = tokenizer.decode([next_token_id])
    return {"input": input.text, "next_word": next_word.strip()}
