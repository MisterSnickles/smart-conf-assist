from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import chromadb
import asyncio
import json
from pathlib import Path
import ollama
from pydantic import BaseModel

app = FastAPI(title="Smart Conference Assistant API")

# Allow the local HTML file to talk to this local server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],
)

# Initialize ChromaDB locally
chroma_client = chromadb.PersistentClient(path="./chroma_data")
collection = chroma_client.get_or_create_collection(name="abstracts_collection")

# Load sample data on startup
@app.on_event("startup")
async def load_sample_data():
    """Load sample.json data into ChromaDB on application startup."""
    sample_file = Path("./sample.json")
    if sample_file.exists():
        try:
            with open(sample_file, 'r') as f:
                data = json.load(f)
            
            # Check if data is already loaded (to avoid duplicates on restart)
            existing_count = collection.count()
            if existing_count == 0:
                valid_records = 0
                for i, record in enumerate(data):
                    required_keys = ["title", "authors", "abstract", "conference_name", "conference_year"]
                    
                    if all(key in record for key in required_keys):
                        doc_id = record.get("paper_id", f"sample_record_{i}")
                        authors_str = ", ".join(record["authors"])
                        
                        collection.add(
                            documents=[record["abstract"]],
                            metadatas=[{
                                "title": record["title"],
                                "authors": authors_str,
                                "conference": f"{record['conference_name']} {record['conference_year']}",
                                "track": record.get("track", "Unknown")
                            }],
                            ids=[doc_id]
                        )
                        valid_records += 1
                
                print(f"Loaded {valid_records} records from sample.json into ChromaDB")
        except Exception as e:
            print(f"Error loading sample.json: {e}")

@app.post("/api/ingest")
async def ingest_abstracts(file: UploadFile = File(...)):
    if not file.filename.endswith('.json'):
        raise HTTPException(status_code=400, detail="Only JSON files are supported.")
    
    try:
        content = await file.read()
        data = json.loads(content)
        
        valid_records = 0
        for i, record in enumerate(data):
            # Strict validation updated to match your actual sample data
            required_keys = ["title", "authors", "abstract", "conference_name", "conference_year"]
            
            if all(key in record for key in required_keys):
                
                # Use the paper_id if it exists, otherwise generate one
                doc_id = record.get("paper_id", f"{file.filename}_record_{i}")
                
                # ChromaDB metadata needs to be flat (strings, ints, floats).
                # So, we convert the ["Ava Carter", "Noah Patel"] list into "Ava Carter, Noah Patel"
                authors_str = ", ".join(record["authors"])
                
                # Add to ChromaDB
                collection.add(
                    documents=[record["abstract"]],
                    metadatas=[{
                        "title": record["title"], 
                        "authors": authors_str, 
                        "conference": f"{record['conference_name']} {record['conference_year']}",
                        "track": record.get("track", "Unknown") # Include track if it exists
                    }],
                    ids=[doc_id]
                )
                valid_records += 1
            else:
                print(f"Skipping a record due to missing required fields.")
                
        return {"message": "File processed", "valid_records_ingested": valid_records}
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Pydantic model for search queries
class SearchQuery(BaseModel):
    query: str
    num_results: int = 5

@app.post("/api/search")
async def search_and_respond(search_query: SearchQuery):
    """Search ChromaDB for relevant papers and generate a response using Ollama."""
    try:
        # Search ChromaDB for relevant papers
        results = collection.query(
            query_texts=[search_query.query],
            n_results=search_query.num_results
        )
        
        if not results or not results['documents'] or len(results['documents'][0]) == 0:
            return {"response": "No relevant papers found.", "papers": []}
        
        # Format the retrieved papers for the LLM
        papers_text = ""
        papers_list = []
        
        for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
            papers_text += f"\nPaper {i+1}: {metadata['title']}\n"
            papers_text += f"Authors: {metadata['authors']}\n"
            papers_text += f"Conference: {metadata['conference']}\n"
            papers_text += f"Abstract: {doc}\n"
            
            papers_list.append({
                "title": metadata['title'],
                "authors": metadata['authors'],
                "conference": metadata['conference'],
                "track": metadata.get('track', 'Unknown')
            })
        
        # Generate response using Ollama in a background thread so the FastAPI event loop is not blocked.
        prompt = f"""Based on the following research papers, answer this query: "{search_query.query}"

Papers:
{papers_text}

Provide a concise, informative response that synthesizes the relevant papers."""
        
        response = await asyncio.to_thread(
            lambda: ollama.generate(
                model="mistral",  # Change to "llama2" if you pulled that instead
                prompt=prompt,
                stream=False
            )
        )
        
        return {
            "query": search_query.query,
            "response": response['response'].strip(),
            "papers": papers_list,
            "num_papers_used": len(papers_list)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@app.get("/api/health")
async def health_check():
    """Check if Ollama is running."""
    try:
        # Try to get model info
        response = ollama.list()
        return {"status": "healthy", "ollama_running": True}
    except Exception as e:
        return {"status": "unhealthy", "ollama_running": False, "error": str(e)}