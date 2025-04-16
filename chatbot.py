import os
import json
import glob
import hashlib
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from neo4j import GraphDatabase
from groq import Groq
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")



print("NEO4J_URI:", os.getenv("NEO4J_URI"))
print("NEO4J_USER:", os.getenv("NEO4J_USER"))
print("NEO4J_PASSWORD:", os.getenv("NEO4J_PASSWORD"))

# Initialize components
nlp = spacy.load("en_core_web_sm")
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
groq_client = Groq(api_key=GROQ_API_KEY)
neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)

# Load JSON data
def load_json_data(data_dir="./data"):
    all_data = []
    for file in glob.glob(os.path.join(data_dir, "*.json")):
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for entry in data:
                entry['file'] = os.path.basename(file)
                entry['id'] = hashlib.md5(entry['question'].encode()).hexdigest()[:8]
                all_data.append(entry)
    df = pd.DataFrame(all_data)
    print(f"Loaded {len(df)} JSON entries from {data_dir}")
    return df

# Chunk data
def chunk_data(df):
    chunks = []
    for _, row in df.iterrows():
        text = f"Question: {row['question']}\nAnswer: {row['answer']}"
        if len(text) > 500:
            split_texts = text_splitter.split_text(text)
            for i, split_text in enumerate(split_texts):
                chunks.append({
                    'id': f"{row['id']}_{i}",
                    'text': split_text,
                    'intent': row['intent'],
                    'file': row['file'],
                    'question_id': row['id']
                })
        else:
            chunks.append({
                'id': row['id'],
                'text': text,
                'intent': row['intent'],
                'file': row['file'],
                'question_id': row['id']
            })
    chunks_df = pd.DataFrame(chunks)
    print(f"Created {len(chunks_df)} chunks")
    return chunks_df

# Load into ChromaDB
def load_to_chromadb(chunks_df):
    texts = chunks_df['text'].tolist()
    metadatas = chunks_df[['id', 'intent', 'file', 'question_id']].to_dict('records')
    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embedder,
        metadatas=metadatas,
        collection_name="ragchatbot2",
        persist_directory="./chroma"  # Save to E:\2llms\chroma
    )
    vectorstore.persist()  # Explicitly save
    print(f"Loaded {len(texts)} texts into ChromaDB at ./chroma")
    return vectorstore

# Load into Neo4j
def load_to_neo4j(df):
    def create_entry(tx, entry):
        entities = [ent.text for ent in nlp(entry['answer']).ents if ent.label_ in ['ORG', 'PERSON', 'GPE']]
        query = """
        MERGE (q:Question {id: $id, text: $question})
        MERGE (a:Answer {text: $answer})
        MERGE (i:Intent {name: $intent})
        MERGE (q)-[:HAS_ANSWER]->(a)
        MERGE (q)-[:HAS_INTENT]->(i)
        """
        params = {
            'id': entry['id'],
            'question': entry['question'],
            'answer': entry['answer'],
            'intent': entry['intent']
        }
        for i, entity in enumerate(entities):
            escaped_entity = entity.replace("'", "\\'")
            query += f"""
            MERGE (e{i}:Entity {{name: '{escaped_entity}'}})
            MERGE (q)-[:MENTIONS_ENTITY]->(e{i})
            """
        tx.run(query, **params)

    with neo4j_driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")  # Clear existing data
        session.run("CREATE INDEX question_id_idx IF NOT EXISTS FOR (q:Question) ON (q.id)")
        session.run("CREATE INDEX intent_name_idx IF NOT EXISTS FOR (i:Intent) ON (i.name)")
        session.run("CREATE INDEX entity_name_idx IF NOT EXISTS FOR (e:Entity) ON (e.name)")
        for _, entry in df.iterrows():
            session.execute_write(create_entry, entry)
        print(f"Loaded {len(df)} entries into Neo4j")

# TF-IDF retrieval
def tfidf_retrieval(query, texts, top_k=50):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    query_vec = vectorizer.transform([query])
    scores = (tfidf_matrix * query_vec.T).toarray().flatten()
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return top_indices

# Reranking
def rerank_chunks(query, chunks, top_n=5):
    query_emb = embedder.embed_query(query)
    chunk_embs = embedder.embed_documents([c['text'] for c in chunks])
    scores = [
        np.dot(query_emb, chunk_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(chunk_emb))
        for chunk_emb in chunk_embs
    ]
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)[:top_n]
    return [chunk for chunk, score in ranked], [score for _, score in ranked]

# Neo4j query
def query_neo4j(query, intent=None):
    doc = nlp(query)
    entities = [ent.text for ent in doc.ents]
    cypher = """
    MATCH (q:Question)
    OPTIONAL MATCH (q)-[:HAS_INTENT]->(i:Intent)
    OPTIONAL MATCH (q)-[:MENTIONS_ENTITY]->(e:Entity)
    WHERE ($intent IS NULL OR i.name = $intent)
    """
    if entities:
        cypher += " OR e.name IN $entities"
    cypher += " RETURN q.id, q.text, collect(i.name) as intents, collect(e.name) as entities LIMIT 10"
    with neo4j_driver.session() as session:
        result = session.run(cypher, intent=intent, entities=entities)
        return [
            {
                'id': record['q.id'],
                'text': record['q.text'],
                'intents': record['intents'],
                'entities': record['entities']
            }
            for record in result
        ]

# Query reformulation #### llm - llama3-8b-8192
def reformulate_query(query):
    prompt = f"Reformulate the following query to be more precise for a university FAQ system:\nQuery: {query}\nReformulated:"
    response = groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100
    )
    return response.choices[0].message.content.strip()

# RAG pipeline    #### meta-llama/llama-4-maverick-17b-128e-instruct
def rag_pipeline(query, vectorstore, texts, max_attempts=2):
    attempt = 0
    while attempt < max_attempts:
        # Step 1: Reformulate query
        reformed_query = reformulate_query(query) if attempt > 0 else query
        print(f"Query: {reformed_query}")

        # Step 2: Infer intent
        intent_prompt = f"Given the query: '{reformed_query}', classify the intent (e.g., admission_counselling, programmes):"
        intent_response = groq_client.chat.completions.create(
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            messages=[{"role": "user", "content": intent_prompt}],
            max_tokens=50
        )
        intent = intent_response.choices[0].message.content.strip().lower()

        # Step 3: TF-IDF filtering
        tfidf_indices = tfidf_retrieval(reformed_query, texts)

        # Step 4: Semantic search
        search_results = vectorstore.similarity_search_with_score(
            reformed_query,
            k=20
            # Removed intent filter to broaden matches
        )
        chunks = [{'text': doc.page_content, 'metadata': doc.metadata} for doc, score in search_results]
        print(f"Found {len(chunks)} chunks in ChromaDB search")

        # Step 5: Neo4j context
        kg_results = query_neo4j(reformed_query, intent)
        kg_context = "\n".join([f"Related Q: {res['text']} (Intents: {res['intents']})" for res in kg_results])
        print(f"Found {len(kg_results)} Neo4j results")

        # Step 6: Rerank
        ranked_chunks, scores = rerank_chunks(reformed_query, chunks)
        if not ranked_chunks or max(scores) < 0.6:  # Lowered threshold
            attempt += 1
            continue

        # Step 7: Generate answer  ##########genma2-9b-it
        context = "\n\n".join([chunk['text'] for chunk in ranked_chunks])
        prompt = f"""
        Using the following context and related information, answer the query concisely. Cite the question ID.
        Context:
        {context}
        Related Info:
        {kg_context}
        Query: {reformed_query}
        Answer:
        """
        response = groq_client.chat.completions.create(
            model="gemma2-9b-it",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200
        )
        answer = response.choices[0].message.content.strip()
        sources = [chunk['metadata']['question_id'] for chunk in ranked_chunks]
        return {
            'answer': answer,
            'sources': sources,
            'confidence': max(scores),
            'reformed_query': reformed_query
        }
    return {'answer': "Sorry, I couldn't find a relevant answer.", 'sources': [], 'confidence': 0.0}

# Main
if __name__ == "__main__":
    # Load and process data
    df = load_json_data()
    chunks_df = chunk_data(df)
    vectorstore = load_to_chromadb(chunks_df)
    load_to_neo4j(df)

    # CLI interface
    print("SRM Chatbot Ready! Type 'exit' to quit.")
    while True:
        query = input("Your question: ").strip()
        if query.lower() == 'exit':
            break
        result = rag_pipeline(query, vectorstore, texts=chunks_df['text'].tolist())
        print(f"\nAnswer: {result['answer']}")
        print(f"Sources: {result['sources']}")
        print(f"Confidence: {result['confidence']:.2f}\n")

    neo4j_driver.close()