# smart-conf-assist
Smart Conference Assistant - Peer Review Article AI Search



## Install Prerequisites
#### Make sure that you have Python installed and Ollama installed on the machine your running the program.



## Python
#### Windows
`winget install Python.Python.3.12`


#### Linux
`sudo apt update`
`sudo apt install python3 python3-venv`



## Ollama
#### Windows
Install
`irm https://ollama.com/install.ps1 | iex`


Download Mistral Model
`ollama run mistral`


#### Linux
Install
`curl -fsSL https://ollama.com/install.sh | sh`


Download Mistral Model
`ollama run mistral`



## Installation Steps
### 1. Clone Git Repository
Navigate to your desired parent folder and type the following command, if you do not have git, please follow installation instructions on git's website. Or download the zip file by clicking the green 'Code' button.
`gh repo clone MisterSnickles/smart-conf-assist`


### 2. Create the virtual environment in local repository
#### (Venv or Virtual Environment is a Python feature that allows the creation of isolated directory trees that contain a specific Python interpreter and independent sets of installed packages. So we can install using various libraries using pip, which is in step 3)
`python -m venv venv`


### 3. Activate it
#### If you are on Windows:
`venv\Scripts\activate`


#### If you are on Mac/Linux:
`source venv/bin/activate`

### 4. Install Libraries Using pip
#### (Make sure to be in the same terminal that is using venv, this command installs the libraries in the virtual environment)
`pip install fastapi uvicorn python-multipart chromadb ollama`


### 5. Launch the Program
#### (Using uvicorn, we can launch the main.py file which intializes the ChromaDB and connection to the locally hosted webpage through port 8000)
`python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000`


### 6. Navigate to locally hosted .html file in File Explorer
<img width="755" height="323" alt="image" src="https://github.com/user-attachments/assets/097717f8-2d59-4491-89b5-df214f9497b9" />

