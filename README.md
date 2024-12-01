# Q-A-Streamlit-App-With-Rag-Using-Gemini-Pro
###
An advanced Question Answering App powered by Retrieval-Augmented Generation using Gemini Pro. This app can answer complex questions based on research papers or documents, making it an incredible tool for information retrieval.For this project, I’ve leveraged Langchain and Google Gemini Pro’s powerful generative AI to create a system that not only retrieves information from documents but also generates contextual, accurate answers. By combining document embeddings with advanced search algorithms, this app retrieves the most relevant parts of the document and then uses a language model to generate answers based on that data.



### Step 1: Clone the repository
```bash
git clone https://github.com/JunaidARahat/Q-A-Streamlit-App-With-Rag-Using-Gemini-Pro.git
```

### Step 2- Create a conda environment after opening the repository

```bash
conda create -n rag python=3.10 -y
```

```bash
conda activate rag
```

### Step 3 - Install the requirements
```bash
pip install -r requirements.txt
```

### Step 4 - Add the Google Api Key in .env File
```bash
GOOGLE_API_KEY=""

```

### Step 5 - Run the application
```bash
streamlit run app.py
``` 