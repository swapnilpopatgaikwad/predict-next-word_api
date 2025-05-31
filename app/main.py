from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to Predict Next Word API"}

@app.get("/demo")
def read_root():
    return {"message": "Welcome to Predict Next Word API"}
