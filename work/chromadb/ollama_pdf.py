# ------------ setup environment ------------ 
import os, sys
VENV_PATH = os.environ.get('VIRTUAL_ENV') # virtual environment path
curr_path = os.path.abspath(__file__)
REPO_DIR = "RAG"
REPO_PATH = curr_path[:curr_path.find(REPO_DIR) + len(REPO_DIR)]

ROOT_DIR = REPO_DIR + "_api"
ROOT_PATH = os.path.join(REPO_PATH, "ref/Telco-RAG/Telco-RAG_api")
sys.path.append(ROOT_PATH)
DOCS_PATH = os.path.join(os.getcwd(), "documents")
DB_PATH = os.path.join(os.getcwd(), "chroma_db")

# libraries
from langchain_ollama import OllamaEmbeddings, OllamaLLM
import chromadb, logging
import preheader # import for custom logger

# ------------ configurations ------------ 
logger = logging.getLogger(__name__) # Setup logging, Supress http conn logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# add all the neccessary config params here that can be modified
config = {
    'embed_model'   : "all-minilm", # model used for embedding
    'chat_model'    : "llama3.2", # model used for chat prompt
    'chunk_size'    : 500, # chunk size to each retreival data
    'chunk_overlap' : 25, # no.of char allowed to overlap
    'k_nearest'     : 3, # no. of relavant retreivals required
    'generate'      : False, # to use llm to generate response or not
}

# Configure ChromaDB client
import chromadb

# The default Chroma client is ephemeral, meaning it will not save to disk
#client = chromadb.Client()

# local copy of database on disk
#client = chromadb.PersistentClient(path=DB_PATH)

# run in a seperate terminal: ```chroma run --host 127.0.0.1 -port 8000 --path /content/chroma_db &``` 
# runs chromadb in the host using exist database in given dir [& => runs in background] [port 8000 by default]
client = chromadb.HttpClient(host='127.0.0.1', port=8000)

llm = OllamaLLM(model=config["chat_model"]) # Configure Ollama Model use for chat prompt

# ------------ database setup ------------ 
import numpy as np
class ChromaDBEmbeddingFunction:
    """Custom embedding function for ChromaDB using embeddings from Ollama"""
    def __init__(self, langchain_embeddings):
        self.langchain_embeddings = langchain_embeddings

    def __call__(self, input):
        # Ensure input is always a list
        if isinstance(input, str):
            input = [input]

        # Get embeddings from Ollama (returns list of lists)
        embeddings = self.langchain_embeddings.embed_documents(input)

        # Convert each list-based embedding to a NumPy array
        return [np.array(embedding) for embedding in embeddings]

    def name(self):
        return self.langchain_embeddings.model

# Initialize the embedding function with Ollama embeddings
OLLAMA_URL = "http://localhost:11434"  # Ollama local endpoint
embedFn = OllamaEmbeddings(model=config["embed_model"], base_url=OLLAMA_URL)

# Define a collection for the RAG workflow
collection_name = "kpi_rag_collection"
collection = client.get_or_create_collection(
    name=collection_name,
    metadata={"description": "A collection of KPI documentations for RAG with Ollama"},
    embedding_function=ChromaDBEmbeddingFunction(embedFn)  # Use the custom embedding function
)

# ------------ pre-processing ------------ 
from src.input import read_docx
from pypdf import PdfReader

def read_pdf(file_path):
    """Read and extract text from a PDF file."""
    try:
        pdf = PdfReader(file_path)
        return '\n'.join(page.extract_text() for page in pdf.pages)
    except Exception as e:
        logger.error(f"Failed to read PDF file at {file_path}: {e}. Removing corrupted file.")
        os.remove(file_path) # comment to ignore
        return None

# Process the documents inside the directory given
def process_documents(doc_dir=DOCS_PATH):
    doc_ds = []
    for root, _, files in os.walk(doc_dir):
        for filename in files:
            file_path = os.path.join(root, filename)
            content = None
            if filename.lower().endswith('.docx'):
                content = read_docx(file_path)
            elif filename.lower().endswith('.pdf'):
                content = read_pdf(file_path)
            if content:
                data_dict = {"text": content, "source": filename}
                doc_ds.append(data_dict)
    return doc_ds

import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Prepare Data to be added to chromaDB collection
doc_ds = process_documents()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=config["chunk_size"],
    chunk_overlap=config["chunk_overlap"],
    separators=[
        "([\s,.\-!?\[\]\(\){}\":;<>]+)",  # Use regex group for complex separators
        " ", "\n", "\t", "\r", "\f", "\v" # Mimics word-splitting
        ],
    length_function=len,
)
doc_chunk_txt = []
doc_chunk_ids = []

for doc in doc_ds:
    # Apply some additional pre-processing
    txt = doc["text"]
    sub = re.sub(r'[ \t\r\f\v]*\n[ \t\r\f\v]*', '\n', txt)
    clean_txt = re.sub(r'\n+', '\n', sub).strip()

    # Split text using LangChain's splitter
    chunks = text_splitter.split_text(clean_txt)
    
    logger.info(f"No. of chunks extracted out of f{doc['source']} = {len(chunks)}")
    doc_chunk_txt.extend(chunks)
    doc_chunk_ids.extend([f"{doc['source']}_chunk{i+1}" for i in range(len( chunks))])

# Add documents to the ChromaDB collection.
collection.add(documents=doc_chunk_txt, ids=doc_chunk_ids)

# ------------ RAG funtionalities ------------ 
from src.get_definitions import find_terms_and_abbreviations_in_sentence

def context_retrieval(query, n_results=5): # can pass multiple queries as list as well
    if type(query)==str:
        query = [query]
    results = collection.query(query_texts=query, n_results=n_results, 
                            include=["metadatas", "documents", "distances", "embeddings"])
    # flatten nested list result of each key (assuming single list enveloped)
    for key in results:
        if results[key] is not None:
            results[key] = results[key][0]
    return results

def submit_prompt(prompt:str):
    if config["generate"] == False:
        logger.warning("config.generate is set to false, cannot use llm!")
        return prompt
    logger.info(f"LLM sumbitted with prompt:\n{prompt}")
    return llm.invoke(prompt)

def compare_embeddings(embed1, embed2):
    # Convert embeddings to numpy arrays for dot product calculation
    embedding1_array = np.array(embed1)
    embedding2_array = np.array(embed2)

    # Calculate and return the dot product
    dot_product = np.dot(embedding1_array, embedding2_array)
    return dot_product

def compare_str(embedFn, inp1, inp2):
    """
    Returns the dot product (cosine similarity) of embeddings for two given text strings.

    The dot product value ranges from -1 to 1:
        - Values close to 1: high similarity.
        - Values close to -1: high dissimilarity.
        - Values close to 0: no apparent similarity.

    Parameters:
        embeddings_model: Embeddings model used to generate embeddings.
        inp1: First text string or embedding vector
        inp2: Second text string or embedding vector

    Returns:
        float: Dot product of the embeddings for text1 and text2.
    """
    e1 = inp1; e2 = inp2

    # Get the embeddings for the two text strings
    if (isinstance(inp1, str)):
        e1 = embedFn.embed_query(inp1)
    if (isinstance(inp1, str)):
        e2 = embedFn.embed_query(inp1)

    return compare_embeddings(e1, e2)

# ToDo: add terms and abbreviations context in form of dict here
terms_definitions = {}
abbreviations_definitions = {}

def rag_pipeline(question:str, retrieve_only:bool=False, should_enhance:bool=False):
    query = question

    if not retrieve_only:
        if should_enhance:
            query = submit_prompt(f"Rephrase the question to be clear and concise:\n\'{question}\'")
        
        if terms_definitions != {} or abbreviations_definitions != {}:
            formatted_terms, formatted_abbreviations = find_terms_and_abbreviations_in_sentence(terms_definitions, abbreviations_definitions, query)
            terms = ', '.join(formatted_terms)
            abbreviations = ', '.join(formatted_abbreviations)
        if terms_definitions != {}:
            query = f"{query}\n\nTerms and Definitions:\n{terms}"
        if abbreviations_definitions != {}:
            query = f"{query}\n\nAbbreviations:\n{abbreviations}\n"
        logger.info(f"Final Enhanced version: [llm:{should_enhance} + Terms & Abbreviations] Query of the given question:\n{query}")
    
    retrived_data = context_retrieval(query, config["k_nearest"])
    query_embedding = embedFn.embed_query(query)
    #logger.warning(retrived_data) # comment it

    context = []
    log_msg = []
    for i in range(len(retrived_data["documents"])): 
        cc =f"\nRetrieval {i+1}. This retrieval is performed from the document 3GPP '{retrived_data["ids"][i]}' : \n...{retrived_data["documents"][i]}...\n"
        retrieval_embedding = retrived_data["embeddings"][i]
        mm = f"{i+1}. Source:{retrived_data["ids"][i]}, Similarity.score:{compare_embeddings(query_embedding, retrieval_embedding)}, " + \
        f"Embedding.size:{retrieval_embedding.shape}, MetaData:{retrived_data["metadatas"][i]}, \nDocument:{retrived_data["documents"][i]}\n"
        context.append(cc)
        log_msg.append(mm)
    
    s_context = '\n'.join(context)
    s_log_msg = '\n'.join(log_msg)
    logger.info(f"Top[{config['k_nearest']}] Retrieved content: \n{s_log_msg}")

    response = None
    if not retrieve_only:
        augmented_prompt = f"""Query:\n{query}\nContext:\n{s_context}
>>> Provide all the possible answers to the following question considering your knowledge and the text provided.
Question: {question}
>>> Ensure none of the answers provided contradicts your knowledge.
"""
        logger.info("######## Final Augmentation Prompt Completed ########")
        response = submit_prompt(augmented_prompt)
    
    return response, context


# main() -> conc only on retreival
if __name__ == "__main__":
    # Example usage: Define Question to query the RAG model
    questions = [
        "Describe active E-RABs in a few lines.", 
        "What are the parameters related to QoS parameter?",
        "What are handover related measurements?",
        "What is IP Throughput?",
    ]
    answers = [] # add answers to measure similarity

    for i in range(len(questions)):
        logger.info(f">>> Question: {questions[i]}")
        response, context = rag_pipeline(questions[i], retrieve_only= (not config['generate']), should_enhance=False) # change config
        if response != None:
            logger.info(f"######### ------ Response from LLM Generated ------ #########\n{response}")
            if 0 <= i < len(answers):
                logger.info(f">>> Expected Answer [similarity score w.r.t llm response = {compare_str(answers[i], response)}]: {answers[i]}")