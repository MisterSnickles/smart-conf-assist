from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import arxiv
import os
import chromadb
import asyncio
import json
from pathlib import Path
import ollama
from pydantic import BaseModel

from io import BytesIO

app = FastAPI(title="Smart Conference Assistant API")

# Serve static files (like index.html) from the current directory
app.mount("/static", StaticFiles(directory="."), name="static")

# Allow the local HTML file to talk to this local server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],
)

@app.get("/api/models")
async def get_available_models():
    """Fetch the list of models currently downloaded in Ollama."""
    try:
        models_dict = ollama.list()
        # Extract just the 'model' string (e.g., 'mistral:latest', 'llama3')
        model_names = [m.get('model', m.get('name')) for m in models_dict.get('models', [])]
        return {"models": model_names}
    except Exception as e:
        return {"models": ["mistral"], "error": str(e)}

# Serve the frontend HTML file
@app.get("/")
async def serve_frontend():
    return FileResponse("index.html")

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

class ArxivFetchQuery(BaseModel):
    topic: str
    max_results: int = 5

@app.post("/api/fetch-arxiv")
async def fetch_and_ingest_arxiv(fetch_query: ArxivFetchQuery):
    """Fetch papers directly from ArXiv and load them into ChromaDB."""
    try:
        # We run the network request in a background thread so it doesn't freeze the API
        def get_arxiv_data():
            client = arxiv.Client()
            search = arxiv.Search(
                query=fetch_query.topic,
                max_results=fetch_query.max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate
            )
            return list(client.results(search))

        results = await asyncio.to_thread(get_arxiv_data)
        
        valid_records = 0
        for result in results:
            # Create a unique ID using ArXiv's built-in ID system
            doc_id = f"arxiv_{result.get_short_id()}"
            
            # Check if this paper is already in our database to avoid duplicates
            existing = collection.get(ids=[doc_id])
            if existing and existing['ids']:
                continue 
                
            clean_abstract = result.summary.replace('\n', ' ')
            authors_str = ", ".join([author.name for author in result.authors])
            
            # Inject directly into ChromaDB!
            collection.add(
                documents=[clean_abstract],
                metadatas=[{
                    "title": result.title,
                    "authors": authors_str,
                    "conference": f"ArXiv Pre-print {result.published.year}",
                    "track": "Live Web Fetch"
                }],
                ids=[doc_id]
            )
            valid_records += 1
            
        return {
            "message": "Fetch complete", 
            "total_found": len(results), 
            "new_papers_added": valid_records
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
    year: str = ""
    model: str = "mistral"  # Default model

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
        
        # Filter by year if provided
        filtered_docs = []
        filtered_metadatas = []
        
        for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
            # Extract year from conference string (format: "Conference 2024")
            conference_str = metadata.get('conference', '')
            conference_year = conference_str.split()[-1] if conference_str else ''
            
            # If year filter is provided, only include papers from that year
            if search_query.year:
                if conference_year == search_query.year:
                    filtered_docs.append(doc)
                    filtered_metadatas.append(metadata)
            else:
                # If no year filter, include all results
                filtered_docs.append(doc)
                filtered_metadatas.append(metadata)
        
        # Check if any papers match the year filter
        if not filtered_docs:
            return {"response": f"No papers found from {search_query.year}.", "papers": []}
        
        # Format the retrieved papers for the LLM
        papers_text = ""
        papers_list = []
        
        for i, (doc, metadata) in enumerate(zip(filtered_docs, filtered_metadatas)):
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
        prompt = f"""
            You are a precise document retrieval assistant. Grab papers only from year that is specified in the query.
            Your task is to identify the SINGLE best matching paper from the provided list based on the user's query.

            User Query: "{search_query.query}"
            

            Papers Data:
            {papers_text}

            Instructions:
            1. Analyze all provided papers and select only the one that most closely relates to the query.
            2. If a match is found, your response must follow this exact format:
            Best Matching Paper
               Title: [Insert Title Here]
               Authors: [Insert Authors Here]
               Conference: [Insert Conference Name and Here]
               Abstract: [Insert Abstract Here, Along with brief explaination on why the paper matches (2 to 3 sentences)]
            3. Do not include any introductory text, pleasantries, or explanations.
            4. If no paper is a relevant match, respond exactly: "I cannot provide a sufficient answer based on the provided papers."
            5. Use only the provided text. Do not invent details or use outside knowledge.
            """
        
        # Generate response
        response = await asyncio.to_thread(
            lambda: ollama.generate(
                model=search_query.model, 
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