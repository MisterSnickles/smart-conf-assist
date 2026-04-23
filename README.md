# smart-conf-assist
Smart Conference Assistant - Peer Review Article AI Search



## Installation Instructions

Make sure that you have Python installed:

### Windows
`winget install Python.Python.3.12`

### Linux
`sudo apt update`
`sudo apt install python3 python3-venv`

### 1. Create the virtual environment
#### (Venv or Virtual Environment is a Python feature that allows the creation of isolated directory trees that contain a specific Python interpreter and independent sets of installed packages. So we can install using various libraries using pip, which is in step 3)
`python -m venv venv`


### 2. Activate it
#### If you are on Windows:
`venv\Scripts\activate`


#### If you are on Mac/Linux:
`source venv/bin/activate`

### 3. Install Libraries Using pip
#### (Make sure to be in the same terminal that is using venv, this command installs the libraries in the virtual environment)
`pip install fastapi uvicorn python-multipart chromadb ollama`


### 4. Launch the Program
#### (Using uvicorn, we can launch the main.py file which intializes the ChromaDB and connection to the locally hosted webpage)
`python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000`
