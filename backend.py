from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from chatbot import load_json_data, chunk_data, load_to_chromadb, load_to_neo4j, rag_pipeline
from neo4j import GraphDatabase
import os

# Initialize FastAPI app
app = FastAPI()

# Initialize Neo4j connection
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Load and prepare data
print("Loading data...")
df = load_json_data()
chunks_df = chunk_data(df)
vectorstore = load_to_chromadb(chunks_df)
load_to_neo4j(df)
texts = chunks_df["text"].tolist()
print("Data loaded successfully!")

# Request and Response Models
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: list
    confidence: float
    reformed_query: str

# Health check endpoint
@app.get("/")
def read_root():
    return {"message": "Chatbot API is running!"}

# Chatbot query endpoint
@app.post("/query", response_model=QueryResponse)
def query_chatbot(request: QueryRequest):
    try:
        # Use the RAG pipeline to process the query
        result = rag_pipeline(request.question, vectorstore, texts)
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            confidence=result["confidence"],
            reformed_query=result["reformed_query"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Shutdown Neo4j connection on app shutdown
@app.on_event("shutdown")
def shutdown_event():
    neo4j_driver.close()
    print("Neo4j connection closed.")