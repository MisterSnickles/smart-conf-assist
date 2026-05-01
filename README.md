# smart-conf-assist
Smart Conference Assistant - Peer Review Article AI Search

The Smart Conference Assistant is a Retrieval-Augmented Generation (RAG) tool designed to help researchers find relevant peer-reviewed articles. Unlike keyword search, this uses vector embeddings to understand the context of your query and generates grounded answers using a local LLM.


<img width="600" height="575" alt="image" src="https://github.com/user-attachments/assets/102b1877-ad73-4da7-a61c-41803a10531d" />


## Features

   **Semantic Search**: Finds papers based on meaning and intent, not just exact keywords.

   **Live ArXiv Fetch**: Don't have the data? Fetch the latest research directly from ArXiv and index it instantly.

   **Local & Private**: Runs entirely on your machine using Ollama—no data ever leaves your computer.

   **Metadata Filtering**: Narrow down searches by specific conference years.

   **Grounded AI**: Prevents "hallucinations" by forcing the AI to only answer based on provided paper abstracts.

## System Structure
The system follows a standard RAG (Retrieval-Augmented Generation) pipeline:

   **Ingestion**: JSON data or ArXiv results are processed.

   **Vectorization**: Abstracts are converted into mathematical vectors via ChromaDB.

   **Retrieval**: User queries are matched against the most relevant abstracts.

   **Augmentation & Generation**: The best matches are sent to Ollama (Phi-3/Mistral) to generate a final summary.

## Install Prerequisites
#### Make sure that you have Python, Git, and Ollama installed on the machine your running the program.


## Python
### Windows (Powershell)
`winget install Python.Python.3.12`


### Linux
`sudo apt update`

`sudo apt install python3 python3-venv`



## Ollama
### Windows (Powershell)
Install
`irm https://ollama.com/install.ps1 | iex`


Download Model (_phi3 is a lightweight option_)
`ollama run mistral`

or

`ollama pull phi3`


### Linux
Install
`curl -fsSL https://ollama.com/install.sh | sh`

Download Model (_phi3 is a lightweight option_)
`ollama run mistral`

or

`ollama pull phi3`



## Installation Steps
## Clone the repo

`git clone https://github.com/MisterSnickles/smart-conf-assist.git`

`cd smart-conf-assist`


## Create and activate virtual environment
### Windows:
`python -m venv venv`

`.\venv\Scripts\activate`

### Linux/Mac:
`python3 -m venv venv`

`source venv/bin/activate`


### 4. Install Libraries Using pip
#### (Make sure to be in the same terminal that is using venv, this command installs the libraries in the virtual environment)
`pip install fastapi uvicorn python-multipart chromadb ollama arxiv`


### 5. Launch the Program
#### (Using uvicorn, we can launch the main.py file which intializes the ChromaDB and connection to the locally hosted webpage through port 8000)
`python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000`


### 6. Access the Application
Open a browser and search `http://127.0.0.1:8000`

