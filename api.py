from fastapi import FastAPI
from pydantic import BaseModel
from rag import RAG
from LLM import generate_response

app = FastAPI()

rag = RAG()

class User_prompt(BaseModel):
    user_prompt: str


@app.post("/generate")
def generate_tour_recommendations(request: User_prompt):
    result = rag.generate(query=request.user_prompt)
    return {"response": result}

@app.post("/query")
def query_tour(request: User_prompt):
    results = rag.query(request.user_prompt)
    return results

@app.post("/generate", response_model=User_prompt)
def generate_response_endpoint(request: User_prompt):
    response = generate_response(request.user_prompt)
    return {"user_prompt": response}