from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import chromadb
import json

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